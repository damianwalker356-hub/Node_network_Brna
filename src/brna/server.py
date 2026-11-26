"""Local web/API host for BRNA Studio and its self-trained text model."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from brna.conversation import ConversationMemory
from brna.language_model import MAX_TRAINING_CHARACTERS, CharacterLanguageModel

LOGGER = logging.getLogger("brna.server")
MAX_REQUEST_BYTES = MAX_TRAINING_CHARACTERS * 4 + 16_384
MAX_PROMPT_CHARACTERS = 4_000
MAX_RESPONSE_TOKENS = 512


class StudioService:
    """Owns the local conversation memory and trainable language model."""

    def __init__(self, data_directory: Path, web_directory: Path | None = None) -> None:
        self.data_directory = data_directory.expanduser().resolve()
        self.data_directory.mkdir(parents=True, exist_ok=True)
        self.web_directory = (web_directory or _web_directory()).resolve()
        if not (self.web_directory / "index.html").is_file():
            raise FileNotFoundError(f"BRNA Studio browser assets were not found in {self.web_directory}")
        self.memory_path = self.data_directory / "conversation-memory.json"
        self.model_path = self.data_directory / "language-model.npz"
        self.lock = threading.RLock()
        self.memory = self._load_memory()
        self.model = (
            CharacterLanguageModel.load(self.model_path)
            if self.model_path.exists()
            else CharacterLanguageModel(hidden_size=32, seed=7)
        )

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {
                "service": "brna-studio",
                "ready": True,
                "model": {
                    "trained": self.model.trained_characters > 0,
                    "training_characters": self.model.trained_characters,
                    "hidden_size": self.model.hidden_size,
                    "vocabulary": "Unicode characters",
                },
                "conversation_examples": len(self.memory.examples),
            }

    def teach(self, prompt: str, response: str) -> dict[str, Any]:
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise TypeError("prompt and response must be strings")
        if not prompt.strip() or not response.strip():
            raise ValueError("both a prompt and response are required")
        if len(prompt) > MAX_PROMPT_CHARACTERS or len(response) > MAX_PROMPT_CHARACTERS * 2:
            raise ValueError("prompt or response is too long")
        with self.lock:
            self.memory.teach(prompt, response)
            self._save_memory()
            return {"saved": True, "conversation_examples": len(self.memory.examples)}

    def train(self, payload: dict[str, Any]) -> dict[str, Any]:
        epochs = payload.get("epochs", 3)
        learning_rate = payload.get("learning_rate", 0.05)
        sequence_length = payload.get("sequence_length", 96)
        if "text" in payload:
            text = payload["text"]
        elif payload.get("source") == "examples":
            with self.lock:
                text = "\n".join(
                    f"User: {example.prompt}\nAssistant: {example.response}"
                    for example in self.memory.examples
                )
        else:
            raise ValueError("provide training text or set source to 'examples'")
        if not isinstance(text, str):
            raise TypeError("training text must be a string")
        with self.lock:
            losses = self.model.train(
                text,
                epochs=epochs,
                learning_rate=learning_rate,
                sequence_length=sequence_length,
            )
            self.model.save(self.model_path)
            return {
                "trained": True,
                "training_characters": self.model.trained_characters,
                "epochs": len(losses),
                "losses": losses,
            }

    def chat(self, prompt: str, *, max_new_tokens: int = 240) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise ValueError(f"prompt must not exceed {MAX_PROMPT_CHARACTERS} characters")
        with self.lock:
            match = self.memory.retrieve(prompt, min_similarity=0.75)
            if match:
                response, similarity = match
                return {"response": response, "source": "taught-example", "similarity": similarity}
            response = self.model.generate(
                f"User: {prompt}\nAssistant:",
                max_new_tokens=max_new_tokens,
            )
            if not response:
                raise RuntimeError("The model generated an empty response; train it with more text and try again.")
            return {"response": response, "source": "local-model"}

    def _load_memory(self) -> ConversationMemory:
        if not self.memory_path.exists():
            return ConversationMemory()
        try:
            return ConversationMemory.from_dict(json.loads(self.memory_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"could not load saved conversation memory: {exc}") from exc

    def _save_memory(self) -> None:
        encoded = json.dumps(self.memory.to_dict(), ensure_ascii=False, allow_nan=False)
        _atomic_write(self.memory_path, encoded.encode("utf-8"))


class StudioRequestHandler(SimpleHTTPRequestHandler):
    """Serve the browser app and bounded JSON endpoints from one local origin."""

    service: StudioService

    def __init__(self, *args: Any, service: StudioService, **kwargs: Any) -> None:
        self.service = service
        super().__init__(*args, directory=str(service.web_directory), **kwargs)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            self._send_json(200, self.service.health())
        elif path == "/api/examples":
            with self.service.lock:
                self._send_json(200, self.service.memory.to_dict())
        elif path.startswith("/api/"):
            self._send_json(404, {"error": "API endpoint not found"})
        else:
            super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {"/api/chat", "/api/teach", "/api/train"}:
            self._send_json(404, {"error": "API endpoint not found"})
            return
        try:
            payload = self._read_json()
            if path == "/api/chat":
                result = self.service.chat(
                    payload.get("prompt"),
                    max_new_tokens=payload.get("max_new_tokens", 240),
                )
            elif path == "/api/teach":
                result = self.service.teach(payload.get("prompt"), payload.get("response"))
            else:
                result = self.service.train(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": f"request body must be valid UTF-8 JSON: {exc}"})
        except (TypeError, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
        except RuntimeError as exc:
            self._send_json(409, {"error": str(exc)})
        else:
            self._send_json(200, result)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length header is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Content-Length must be a whole number") from exc
        if not 1 <= length <= MAX_REQUEST_BYTES:
            raise ValueError(f"request body must be between 1 and {MAX_REQUEST_BYTES} bytes")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise TypeError("request body must be a JSON object")
        return data

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)


def create_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    data_directory: Path | None = None,
    web_directory: Path | None = None,
) -> ThreadingHTTPServer:
    service = StudioService(data_directory or _default_data_directory(), web_directory)

    class BoundHandler(StudioRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, service=service, **kwargs)

    return ThreadingHTTPServer((host, port), BoundHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the self-hosted BRNA Studio and local language model.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (defaults to localhost only)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (defaults to 8765)")
    parser.add_argument("--data-dir", type=Path, help="model and conversation storage directory")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_directory = (args.data_dir or _default_data_directory()).expanduser().resolve()
    server = create_server(host=args.host, port=args.port, data_directory=data_directory)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"BRNA Studio is running at {url}")
    print(f"Local model data is stored in {data_directory}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BRNA Studio.")
    finally:
        server.server_close()


def _web_directory() -> Path:
    bundle_root = getattr(sys, "_MEIPASS", None)
    candidates = []
    if bundle_root:
        candidates.append(Path(bundle_root) / "web")
    candidates.extend(
        (
            Path(__file__).resolve().parents[2] / "web",
            Path(sys.prefix) / "share" / "brna-studio" / "web",
        )
    )
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    raise FileNotFoundError("BRNA Studio web assets were not found in the app bundle or Python installation")


def _default_data_directory() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return root / "BRNA Studio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "BRNA Studio"
    root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "brna-studio"


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except OSError:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


if __name__ == "__main__":
    main()
