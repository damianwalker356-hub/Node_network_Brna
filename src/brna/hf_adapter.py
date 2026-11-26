"""Optional local-only Hugging Face causal-LM backend with per-node LoRA adapters."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

MAX_BACKBONE_TRAINING_CHARACTERS = 50_000
MAX_BACKBONE_SEQUENCE_LENGTH = 2_048
MAX_BACKBONE_OUTPUT_TOKENS = 512
_ADAPTER_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class LocalLoRABackbone:
    """Share one local frozen causal LM while keeping trainable adapters per node."""

    def __init__(
        self,
        model_directory: str | Path,
        adapter_directory: str | Path,
        *,
        device: str = "auto",
        lora_rank: int = 8,
    ) -> None:
        self.model_directory = Path(model_directory).expanduser().resolve()
        self.adapter_directory = Path(adapter_directory).expanduser().resolve()
        if not self.model_directory.is_dir():
            raise FileNotFoundError(
                f"local pretrained model directory was not found: {self.model_directory}"
            )
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        if isinstance(lora_rank, bool) or not isinstance(lora_rank, int) or not 1 <= lora_rank <= 64:
            raise ValueError("lora_rank must be a whole number from 1 to 64")
        try:
            import torch
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local pretrained models require the optional dependencies; install with "
                "`pip install -e '.[llm]'`."
            ) from exc

        self._torch = torch
        self._LoraConfig = LoraConfig
        self._TaskType = TaskType
        self._get_peft_model = get_peft_model
        self.device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        dtype = torch.float32
        if self.device == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_directory),
            local_files_only=True,
        )
        if self.tokenizer.eos_token_id is None:
            raise ValueError("the local tokenizer must define an EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_directory),
            local_files_only=True,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.lora_rank = lora_rank
        self.adapter_names: dict[str, str] = {}
        self._trained_adapters: set[str] = set()
        self._target_modules: list[str] | None = None

    @property
    def hidden_size(self) -> int:
        config = self.model.config
        value = getattr(config, "hidden_size", None) or getattr(
            config, "n_embd", None
        )
        if not isinstance(value, int) or value <= 0:
            raise ValueError("the local model config does not expose a supported hidden size")
        return value

    def ensure_adapter(self, adapter_id: str, role: str) -> str:
        if not isinstance(adapter_id, str) or not _ADAPTER_ID.fullmatch(adapter_id):
            raise ValueError("adapter_id must contain 1 to 64 letters, digits, '_' or '-'")
        if not isinstance(role, str) or not role.strip() or len(role) > 240:
            raise ValueError("role must be a non-empty string of at most 240 characters")
        if adapter_id in self.adapter_names:
            return self.adapter_names[adapter_id]
        adapter_name = f"node_{adapter_id}"
        config = self._LoraConfig(
            task_type=self._TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_rank * 2,
            lora_dropout=0.05,
            target_modules=self._select_target_modules(),
        )
        saved_adapter = self.adapter_directory / adapter_id
        if not hasattr(self.model, "peft_config"):
            self.model = self._get_peft_model(
                self.model,
                config,
                adapter_name="brna_bootstrap",
            )
        if saved_adapter.is_dir():
            self.model.load_adapter(
                str(saved_adapter),
                adapter_name=adapter_name,
                is_trainable=True,
            )
            self._trained_adapters.add(adapter_id)
        else:
            self.model.add_adapter(adapter_name, config)
        self.adapter_names[adapter_id] = adapter_name
        self.model.eval()
        return adapter_name

    def generate(
        self,
        prompt: str,
        *,
        adapter_id: str | None = None,
        role: str | None = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        soft_prefix: Any | None = None,
    ) -> str:
        self._validate_prompt(prompt)
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or not 1 <= max_new_tokens <= MAX_BACKBONE_OUTPUT_TOKENS
        ):
            raise ValueError(
                f"max_new_tokens must be a whole number from 1 to {MAX_BACKBONE_OUTPUT_TOKENS}"
            )
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 < temperature <= 2
        ):
            raise ValueError("temperature must be finite and in (0, 2]")
        if adapter_id is not None:
            if role is None:
                raise ValueError("role is required when generating with a node adapter")
            adapter_name = self.ensure_adapter(adapter_id, role)
            self.model.set_adapter(adapter_name)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_BACKBONE_SEQUENCE_LENGTH,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]
        if soft_prefix is not None:
            prompt_embeddings = self.model.get_input_embeddings()(
                encoded["input_ids"]
            )
            prefix = soft_prefix.to(
                device=self.device,
                dtype=prompt_embeddings.dtype,
            )
            if (
                prefix.ndim != 3
                or prefix.shape[0] != 1
                or prefix.shape[2] != prompt_embeddings.shape[2]
                or not self._torch.isfinite(prefix).all()
            ):
                raise ValueError(
                    "soft_prefix must be a finite (1, prefix_tokens, model_hidden_size) tensor"
                )
            encoded["inputs_embeds"] = self._torch.cat(
                (prefix, prompt_embeddings),
                dim=1,
            )
            encoded["attention_mask"] = self._torch.ones(
                encoded["inputs_embeds"].shape[:2],
                dtype=self._torch.long,
                device=self.device,
            )
        with self._torch.inference_mode():
            generation_inputs = (
                {"inputs_embeds": encoded["inputs_embeds"], "attention_mask": encoded["attention_mask"]}
                if soft_prefix is not None
                else encoded
            )
            generated = self.model.generate(
                **generation_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            generated[0] if soft_prefix is not None else generated[0, prompt_length:],
            skip_special_tokens=True,
        ).strip()

    def encode_context_tensor(self, text: str, *, adapter_id: str, role: str):
        self._validate_prompt(text)
        adapter_name = self.ensure_adapter(adapter_id, role)
        self.model.set_adapter(adapter_name)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_BACKBONE_SEQUENCE_LENGTH,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.no_grad():
            result = self.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        return result.hidden_states[-1][0, -1].float()

    def stack_contexts(self, contexts: list[Any]):
        return self._torch.stack(contexts)

    def tensor_to_numpy(self, values) -> np.ndarray:
        return values.detach().float().cpu().numpy()

    def create_soft_prefix_graph(
        self,
        *,
        node_count: int,
        adjacency: np.ndarray,
        latent_size: int,
        hops: int,
        seed: int,
    ):
        torch = self._torch
        adjacency_values = np.asarray(adjacency, dtype=float)
        if (
            adjacency_values.shape != (node_count, node_count)
            or not np.isin(adjacency_values, (0.0, 1.0)).all()
            or np.any(np.diag(adjacency_values))
        ):
            raise ValueError("soft-prefix graph adjacency is invalid")
        if not 1 <= latent_size <= 256:
            raise ValueError("soft-prefix graph latent_size must be from 1 to 256")
        if not 0 <= hops <= 16:
            raise ValueError("soft-prefix graph hops must be from 0 to 16")
        sources, targets = np.nonzero(adjacency_values)
        source_tensor = torch.as_tensor(sources, dtype=torch.long, device=self.device)
        target_tensor = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        hidden_size = self.hidden_size
        device = self.device

        class SoftPrefixGraph(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.node_count = node_count
                self.latent_size = latent_size
                self.hops = hops
                self.hidden_size = hidden_size
                self.input_projection = torch.nn.Linear(hidden_size, latent_size)
                self.self_projection = torch.nn.Linear(latent_size, latent_size)
                self.message_projection = torch.nn.Linear(latent_size, latent_size, bias=False)
                self.prefix_projection = torch.nn.Linear(
                    latent_size,
                    hidden_size,
                )
                self.node_embeddings = torch.nn.Parameter(
                    torch.randn(node_count, latent_size) * 0.02
                )
                self.edge_gates = torch.nn.Parameter(
                    torch.zeros(len(sources))
                )
                self.register_buffer("edge_sources", source_tensor)
                self.register_buffer("edge_targets", target_tensor)
                self.to(device)

            def forward(self, context_states):
                if (
                    context_states.shape != (node_count, hidden_size)
                    or not torch.isfinite(context_states).all()
                ):
                    raise ValueError(
                        f"context_states must have shape ({node_count}, {hidden_size}) "
                        "and contain finite values"
                    )
                states = torch.tanh(
                    self.input_projection(context_states)
                    + self.node_embeddings
                )
                for _ in range(self.hops):
                    aggregate = torch.zeros_like(states)
                    if self.edge_sources.numel():
                        gates = torch.sigmoid(self.edge_gates)
                        messages = states[self.edge_sources] * gates.unsqueeze(-1)
                        aggregate = aggregate.index_add(
                            0,
                            self.edge_targets,
                            messages / max(1, node_count),
                        )
                    states = torch.tanh(
                        self.self_projection(states)
                        + self.message_projection(aggregate)
                    )
                    norms = torch.linalg.vector_norm(states, dim=-1, keepdim=True)
                    states = states / torch.clamp(norms, min=1.0)
                prefixes = self.prefix_projection(states).unsqueeze(1)
                prefix_norms = torch.linalg.vector_norm(
                    prefixes,
                    dim=-1,
                    keepdim=True,
                )
                prefixes = prefixes / torch.clamp(prefix_norms, min=1.0)
                return {
                    "latents": states,
                    "prefix_embeddings": prefixes,
                }

        cuda_devices = (
            [torch.cuda.current_device()]
            if device.startswith("cuda") and torch.cuda.is_available()
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            return SoftPrefixGraph()

    def encode_context(
        self,
        text: str,
        *,
        adapter_id: str,
        role: str,
    ) -> np.ndarray:
        self._validate_prompt(text)
        adapter_name = self.ensure_adapter(adapter_id, role)
        self.model.set_adapter(adapter_name)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_BACKBONE_SEQUENCE_LENGTH,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            result = self.model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        last_token = result.hidden_states[-1][0, -1].float().cpu().numpy()
        if not np.isfinite(last_token).all():
            raise RuntimeError("local model returned a non-finite hidden state")
        return last_token

    def train_text(
        self,
        adapter_id: str,
        role: str,
        text: str,
        *,
        epochs: int = 1,
        learning_rate: float = 2e-4,
    ) -> list[float]:
        self._validate_training_text(text, epochs, learning_rate)
        adapter_name = self.ensure_adapter(adapter_id, role)
        self.model.set_adapter(adapter_name)
        encoded = self.tokenizer(
            f"Role: {role}\nTraining material:\n{text}",
            return_tensors="pt",
        )
        all_ids = encoded["input_ids"]
        chunks = [
            all_ids[
                :, start : start + MAX_BACKBONE_SEQUENCE_LENGTH
            ].to(self.device)
            for start in range(
                0,
                all_ids.shape[1] - 1,
                MAX_BACKBONE_SEQUENCE_LENGTH - 1,
            )
        ]
        if not chunks:
            raise ValueError("training text must encode to at least two tokens")
        parameters = [item for item in self.model.parameters() if item.requires_grad]
        if not parameters:
            raise RuntimeError(f"LoRA adapter {adapter_name!r} has no trainable parameters")
        optimizer = self._torch.optim.AdamW(parameters, lr=float(learning_rate))
        losses = []
        self.model.train()
        try:
            for _ in range(epochs):
                epoch_losses = []
                for input_ids in chunks:
                    optimizer.zero_grad(set_to_none=True)
                    result = self.model(
                        input_ids=input_ids,
                        labels=input_ids,
                        use_cache=False,
                    )
                    loss = result.loss
                    if not self._torch.isfinite(loss):
                        raise RuntimeError("local model produced a non-finite training loss")
                    loss.backward()
                    self._torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                    optimizer.step()
                    epoch_losses.append(float(loss.detach().cpu()))
                losses.append(sum(epoch_losses) / len(epoch_losses))
        finally:
            self.model.eval()
        self._save_adapter(adapter_id, adapter_name)
        self._trained_adapters.add(adapter_id)
        return losses

    def reinforce(
        self,
        adapter_id: str,
        role: str,
        prompt: str,
        completion: str,
        advantage: float,
        *,
        learning_rate: float = 1e-5,
        soft_prefix=None,
        graph_parameters=(),
    ) -> float:
        self._validate_prompt(prompt)
        if not isinstance(completion, str) or not completion.strip():
            raise ValueError("completion must be a non-empty string")
        if (
            isinstance(advantage, bool)
            or not isinstance(advantage, (int, float))
            or not math.isfinite(advantage)
            or not -1 <= advantage <= 1
        ):
            raise ValueError("advantage must be finite and in [-1, 1]")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1e-2
        ):
            raise ValueError("learning_rate must be finite and in (0, 0.01]")
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_BACKBONE_SEQUENCE_LENGTH,
        )["input_ids"]
        completion_ids = self.tokenizer(
            completion,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_BACKBONE_SEQUENCE_LENGTH,
        )["input_ids"]
        if completion_ids.shape[1] == 0:
            raise ValueError("completion must encode to at least one token")
        if prompt_ids.shape[1] + completion_ids.shape[1] > MAX_BACKBONE_SEQUENCE_LENGTH:
            prompt_ids = prompt_ids[
                :, : MAX_BACKBONE_SEQUENCE_LENGTH - completion_ids.shape[1]
            ]
        input_ids = self._torch.cat((prompt_ids, completion_ids), dim=1).to(self.device)
        prefix = (
            self._torch.empty((1, 0, self.hidden_size), device=self.device)
            if soft_prefix is None
            else soft_prefix.to(device=self.device)
        )
        prompt_length = prompt_ids.shape[1]
        embeddings = self.model.get_input_embeddings()(input_ids)
        combined_embeddings = self._torch.cat(
            (prefix.to(dtype=embeddings.dtype), embeddings),
            dim=1,
        )
        attention_mask = self._torch.ones(
            combined_embeddings.shape[:2],
            dtype=self._torch.long,
            device=self.device,
        )
        adapter_name = self.ensure_adapter(adapter_id, role)
        self.model.set_adapter(adapter_name)
        parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and adapter_name in name
        ]
        parameters.extend(
            parameter
            for parameter in graph_parameters
            if parameter.requires_grad
        )
        if not parameters:
            raise RuntimeError(f"LoRA adapter {adapter_name!r} has no trainable parameters")
        optimizer = self._torch.optim.AdamW(parameters, lr=float(learning_rate))
        self.model.train()
        try:
            optimizer.zero_grad(set_to_none=True)
            result = self.model(
                inputs_embeds=combined_embeddings,
                attention_mask=attention_mask,
                use_cache=False,
            )
            start = prefix.shape[1] + prompt_length - 1
            end = start + completion_ids.shape[1]
            logits = result.logits[:, start:end, :]
            targets = completion_ids.to(self.device)
            log_probs = self._torch.log_softmax(logits, dim=-1)
            selected = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            objective = -float(advantage) * selected.mean()
            if not self._torch.isfinite(objective):
                raise RuntimeError("local model produced a non-finite policy loss")
            objective.backward()
            self._torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            policy_loss = float(objective.detach().cpu())
        finally:
            self.model.eval()
        self._save_adapter(adapter_id, adapter_name)
        self._trained_adapters.add(adapter_id)
        return policy_loss

    def grpo_group_update(
        self,
        adapter_id: str,
        role: str,
        prompt: str,
        completions: list[str],
        rewards: list[float],
        *,
        soft_prefix,
        graph_parameters,
        learning_rate: float = 1e-5,
        kl_beta: float = 0.02,
    ) -> dict[str, float]:
        self._validate_prompt(prompt)
        if not isinstance(completions, list) or not 2 <= len(completions) <= 8:
            raise ValueError("GRPO requires a group of 2 to 8 completions")
        if not isinstance(rewards, list) or len(rewards) != len(completions):
            raise ValueError("one finite reward is required per completion")
        if any(
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
            for score in rewards
        ):
            raise ValueError("GRPO verifier rewards must be finite values in [0, 1]")
        if (
            isinstance(kl_beta, bool)
            or not isinstance(kl_beta, (int, float))
            or not math.isfinite(kl_beta)
            or not 0 <= kl_beta <= 1
        ):
            raise ValueError("kl_beta must be finite and in [0, 1]")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1e-3
        ):
            raise ValueError("GRPO learning_rate must be finite and in (0, 0.001]")

        scores = self._torch.as_tensor(rewards, device=self.device, dtype=self._torch.float32)
        advantages = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-6)
        advantages = advantages.clamp(-5.0, 5.0)
        adapter_name = self.ensure_adapter(adapter_id, role)
        self.model.set_adapter(adapter_name)
        parameters = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and adapter_name in name
        ]
        parameters.extend(
            parameter
            for parameter in graph_parameters
            if parameter.requires_grad
        )
        if not parameters:
            raise RuntimeError(f"LoRA adapter {adapter_name!r} has no trainable parameters")
        optimizer = self._torch.optim.AdamW(parameters, lr=float(learning_rate))
        losses = []
        kls = []
        self.model.train()
        try:
            optimizer.zero_grad(set_to_none=True)
            for index, completion in enumerate(completions):
                if not isinstance(completion, str) or not completion.strip():
                    raise ValueError("each GRPO completion must be non-empty text")
                objective, kl = self._grpo_objective(
                    prompt,
                    completion,
                    soft_prefix,
                    advantages[index],
                    float(kl_beta),
                )
                losses.append(objective)
                kls.append(kl)
            total = self._torch.stack(losses).mean()
            if not self._torch.isfinite(total):
                raise RuntimeError("local model produced a non-finite GRPO objective")
            total.backward()
            self._torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
        finally:
            self.model.eval()
        self._save_adapter(adapter_id, adapter_name)
        self._trained_adapters.add(adapter_id)
        return {
            "loss": float(total.detach().cpu()),
            "mean_kl": float(self._torch.stack(kls).mean().detach().cpu()),
            "mean_reward": float(scores.mean().detach().cpu()),
            "reward_std": float(scores.std(unbiased=False).detach().cpu()),
            "group_size": len(completions),
        }

    def _grpo_objective(
        self,
        prompt: str,
        completion: str,
        soft_prefix,
        advantage,
        kl_beta: float,
    ):
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_BACKBONE_SEQUENCE_LENGTH,
        )["input_ids"]
        completion_ids = self.tokenizer(
            completion,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_BACKBONE_SEQUENCE_LENGTH,
        )["input_ids"]
        if completion_ids.shape[1] == 0:
            raise ValueError("completion must encode to at least one token")
        if prompt_ids.shape[1] + completion_ids.shape[1] > MAX_BACKBONE_SEQUENCE_LENGTH:
            prompt_ids = prompt_ids[
                :, : MAX_BACKBONE_SEQUENCE_LENGTH - completion_ids.shape[1]
            ]
        token_ids = self._torch.cat((prompt_ids, completion_ids), dim=1).to(self.device)
        prefix = soft_prefix.to(device=self.device)
        token_embeddings = self.model.get_input_embeddings()(token_ids)
        inputs_embeds = self._torch.cat(
            (prefix.to(dtype=token_embeddings.dtype), token_embeddings),
            dim=1,
        )
        attention_mask = self._torch.ones(
            inputs_embeds.shape[:2],
            dtype=self._torch.long,
            device=self.device,
        )
        start = prefix.shape[1] + prompt_ids.shape[1] - 1
        end = start + completion_ids.shape[1]
        current = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, start:end, :]
        with self._torch.no_grad(), self.model.disable_adapter():
            reference = self.model(
                inputs_embeds=inputs_embeds.detach(),
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[:, start:end, :]
        current_log = self._torch.log_softmax(current.float(), dim=-1)
        reference_log = self._torch.log_softmax(reference.float(), dim=-1)
        target_ids = completion_ids.to(self.device)
        token_log_probability = current_log.gather(
            -1,
            target_ids.unsqueeze(-1),
        ).squeeze(-1).mean()
        current_probability = current_log.exp()
        kl = (
            current_probability * (current_log - reference_log)
        ).sum(dim=-1).mean()
        objective = -advantage * token_log_probability + kl_beta * kl
        return objective, kl.detach()

    def has_trained_adapter(self, adapter_id: str) -> bool:
        return adapter_id in self._trained_adapters or (
            (self.adapter_directory / adapter_id / "adapter_config.json").is_file()
            and any(
                (self.adapter_directory / adapter_id / filename).is_file()
                for filename in ("adapter_model.safetensors", "adapter_model.bin")
            )
        )

    def _select_target_modules(self) -> list[str]:
        if self._target_modules is not None:
            return self._target_modules
        suffixes = {
            name.rsplit(".", 1)[-1]
            for name, _ in self.model.named_modules()
        }
        for candidates in (
            ("q_proj", "v_proj"),
            ("query", "value"),
            ("c_attn",),
            ("Wqkv",),
        ):
            selected = [name for name in candidates if name in suffixes]
            if selected:
                self._target_modules = selected
                return selected
        raise ValueError(
            "no supported attention projection modules were found for LoRA "
            "(tried q_proj/v_proj, query/value, c_attn, and Wqkv)"
        )

    def _save_adapter(self, adapter_id: str, adapter_name: str) -> None:
        target = self.adapter_directory / adapter_id
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(
            str(target),
            selected_adapters=[adapter_name],
        )

    @staticmethod
    def _validate_prompt(prompt: str) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(prompt) > 20_000:
            raise ValueError("prompt must not exceed 20000 characters")

    @staticmethod
    def _validate_training_text(
        text: str,
        epochs: int,
        learning_rate: float,
    ) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("training text must be a non-empty string")
        if len(text) > MAX_BACKBONE_TRAINING_CHARACTERS:
            raise ValueError(
                f"training text must not exceed {MAX_BACKBONE_TRAINING_CHARACTERS} characters"
            )
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be a whole number from 1 to 50")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or not 0 < learning_rate <= 1e-2
        ):
            raise ValueError("learning_rate must be finite and in (0, 0.01]")
