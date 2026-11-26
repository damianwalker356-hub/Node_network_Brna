import {
  DEFAULT_DATASET,
  DenseNetwork,
  MessagePassingNetwork,
  createNetworkFromJSON,
  evaluateBinaryAccuracy,
  evolveGeneration,
  initialPopulation,
  policyStep,
  retrieveReply,
  validateArchitecture,
} from "./core.js";
import { LocalCharacterModel } from "./local-model.js";

const STORAGE_KEY = "brna-studio-v1";
const LOCAL_TEXT_MODEL_STORAGE_KEY = "brna-local-text-model-v1";
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const navLinks = $$(".nav-link");
const pages = $$("[data-page-panel]");
const layerList = $("#layerList");
const toast = $("#toast");

let network;
let modelName = "BRNA prototype";
let models = [];
let activeModelId = "";
let customDataset = null;
let examples = [];
let policy = { baseline: 0, updates: 0, pending: null };
let toastTimer;
let installPrompt;
let localModelHealth = null;
let localAgentNetwork = null;
let localAgentGraph = null;
let localAgentGraphError = null;
let reasoningTaskCatalog = null;
let localCharacterModel = new LocalCharacterModel();

function restoreLocalCharacterModel() {
  try {
    const saved = localStorage.getItem(LOCAL_TEXT_MODEL_STORAGE_KEY);
    if (saved) localCharacterModel = LocalCharacterModel.fromJSON(JSON.parse(saved));
  } catch (error) {
    console.error("Could not restore the browser-local text model", error);
    $("#modelTrainingStatus").textContent =
      `Saved browser model could not be restored: ${error.message}`;
  }
}

function persistLocalCharacterModel() {
  localStorage.setItem(LOCAL_TEXT_MODEL_STORAGE_KEY, JSON.stringify(localCharacterModel));
}

function loadWorkspace() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null");
    if (!saved) {
      network = new MessagePassingNetwork({}, { seed: 7 });
      activeModelId = makeId();
      models = [{ id: activeModelId, name: modelName, model: network.toJSON(), policy: { baseline: 0, updates: 0 } }];
      examples = [];
      return;
    }
    models = Array.isArray(saved.models) ? saved.models : [];
    if (models.length === 0 && saved.model) {
      models = [{
        id: makeId(),
        name: typeof saved.name === "string" && saved.name.trim() ? saved.name : modelName,
        model: saved.model,
        policy: saved.policy,
      }];
    }
    if (models.length === 0) {
      network = new MessagePassingNetwork({}, { seed: 7 });
      activeModelId = makeId();
      models.push({ id: activeModelId, name: modelName, model: network.toJSON() });
    } else {
      const selected = models.find((item) => item.id === saved.activeModelId) ?? models[0];
      activeModelId = selected.id;
      network = createNetworkFromJSON(selected.model);
      modelName = selected.name;
      customDataset = Array.isArray(selected.dataset) ? selected.dataset : null;
      if (customDataset) network.loss(customDataset);
      policy = {
        baseline: Number.isFinite(selected.policy?.baseline) ? selected.policy.baseline : 0,
        updates: Number.isInteger(selected.policy?.updates) ? selected.policy.updates : 0,
        pending: null,
      };
    }
    examples = Array.isArray(saved.conversations) ? saved.conversations : [];
  } catch (error) {
    network = new MessagePassingNetwork({}, { seed: 7 });
    activeModelId = makeId();
    customDataset = null;
    policy = { baseline: 0, updates: 0, pending: null };
    models = [{ id: activeModelId, name: modelName, model: network.toJSON() }];
    examples = [];
    announce(`Saved workspace could not be loaded: ${error.message}`);
  }
}

function saveWorkspace() {
  try {
    let activeModel = models.find((item) => item.id === activeModelId);
    if (!activeModel) {
      activeModel = { id: activeModelId, name: modelName };
      models.push(activeModel);
    }
    activeModel.name = modelName;
    activeModel.model = network.toJSON();
    activeModel.dataset = customDataset;
    activeModel.policy = { baseline: policy.baseline, updates: policy.updates };
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        activeModelId,
        models,
        conversations: examples,
      }),
    );
    $("#saveStatus").textContent = "Saved on this device";
  } catch (error) {
    $("#saveStatus").textContent = "Save failed";
    announce(`Could not save this workspace: ${error.message}`);
  }
}

async function localModelRequest(path, payload) {
  const response = await fetch(`./api/${path}`, {
    method: payload ? "POST" : "GET",
    headers: payload ? { "Content-Type": "application/json" } : {},
    body: payload ? JSON.stringify(payload) : undefined,
    cache: "no-store",
  });
  let result;
  try {
    result = await response.json();
  } catch (error) {
    throw new Error(`Local model server returned an invalid response: ${error.message}`);
  }
  if (!response.ok) throw new Error(result.error || `Local model server returned HTTP ${response.status}.`);
  return result;
}

async function refreshLocalModel() {
  try {
    localModelHealth = await localModelRequest("health");
    localAgentNetwork = localModelHealth.agent_network;
    renderAgentNetwork(localAgentNetwork);
    await refreshReasoningTasks();
    let syncError = null;
    try {
      for (const example of examples) {
        await localModelRequest("teach", example);
      }
    } catch (error) {
      syncError = error;
    }
    localModelHealth = await localModelRequest("health");
    localAgentNetwork = localModelHealth.agent_network;
    renderAgentNetwork(localAgentNetwork);
    await refreshAgentGraph();
    const model = localModelHealth.model;
    if (model.backend === "local-pretrained-causal-lm") {
      $("#localModelBadge").textContent = model.trained
        ? "Pretrained model · adapter trained"
        : "Pretrained model ready";
      $("#localModelStatus").textContent =
        `Local pretrained causal model connected · ${model.backbone_hidden_size} hidden units · ` +
        `${model.device} · ${localModelHealth.conversation_examples} saved examples. ` +
        "Model files and adapter training stay on this computer.";
    } else {
      $("#localModelBadge").textContent = model.trained ? "Model trained" : "Model needs training";
      $("#localModelStatus").textContent =
        `Self-hosted scratch model connected · ${model.training_characters.toLocaleString()} training characters · ` +
        `${localModelHealth.conversation_examples} saved examples. ` +
        "The model and data stay on this computer.";
    }
    $("#trainFromExamples").disabled = localModelHealth.conversation_examples === 0;
    $("#trainFromCorpus").disabled = false;
    if (syncError) {
      $("#modelTrainingStatus").textContent =
        `Server connected, but browser examples could not all sync: ${syncError.message}`;
    }
  } catch (error) {
    localModelHealth = null;
    localAgentNetwork = null;
    localAgentGraph = null;
    localAgentGraphError = error.message;
    reasoningTaskCatalog = null;
    renderAgentNetwork(null);
    $("#localModelBadge").textContent = localCharacterModel.trained
      ? "Browser model trained"
      : "Browser model ready";
    $("#localModelStatus").textContent =
      `No local Python server detected. Training and generation run in this browser on this computer; ` +
      `no separate server is needed. ` +
      `${localCharacterModel.trainingCharacters.toLocaleString()} cumulative training characters. ` +
      "This compact character model stays in this browser profile. The optional Python server adds " +
      "the larger RNN and agent features.";
    $("#trainFromExamples").disabled = examples.length === 0;
    $("#trainFromCorpus").disabled = false;
  }
}

function renderAgentNetwork(state) {
  const nodes = state?.nodes ?? [];
  $("#agentNodeCount").textContent = `${nodes.length} node${nodes.length === 1 ? "" : "s"}`;
  const list = $("#agentNodeList");
  list.replaceChildren();
  for (const node of nodes) {
    const item = document.createElement("div");
    item.className = "example-item";
    const details = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = `${node.name} · ${node.hidden_size} hidden units`;
    const role = document.createElement("p");
    role.textContent = node.role;
    details.append(title, role);
    const training = document.createElement("p");
    training.textContent =
      `${node.trained_characters.toLocaleString()} trained characters · ` +
      `${node.reinforcement_updates} reward updates`;
    item.append(details, training);
    list.append(item);
  }
  if (nodes.length === 0) {
    const empty = document.createElement("p");
    empty.className = "field-help";
    empty.textContent = state
      ? "No model nodes created yet."
      : "Agent nodes are available while the local server is running.";
    list.append(empty);
  }
  const connected = Boolean(state);
  const trainedNodes = nodes.filter((node) => node.trained_characters > 0).length;
  $("#createAgentNode").disabled = !connected || nodes.length >= 8;
  $("#trainAgentNetwork").disabled = !connected || nodes.length === 0;
  $("#uploadAgentCorpus").disabled = !connected || nodes.length === 0;
  $("#runAgentNetwork").disabled = !connected || nodes.length < 2 || trainedNodes !== nodes.length;
  $("#trainReasoningDataset").disabled =
    !connected || !reasoningTaskCatalog || nodes.length === 0;
  $("#runVerifiedEpisode").disabled =
    !connected || !reasoningTaskCatalog || nodes.length < 2 || trainedNodes !== nodes.length;
  const grpoEnabled =
    localModelHealth?.model.backend === "local-pretrained-causal-lm" &&
    trainedNodes === nodes.length &&
    nodes.length > 0 &&
    Boolean(reasoningTaskCatalog);
  $("#runGRPOEpisode").disabled = !grpoEnabled;
  if (grpoEnabled) {
    $("#grpoEpisodeStatus").textContent =
      "Verifier-scored candidate groups update each node's LoRA adapter and continuous graph soft prefixes.";
  }
  if (!connected) {
    $("#agentBuilderStatus").textContent = "Connect the local BRNA Studio server to create persistent agent models.";
  } else {
    $("#agentBuilderStatus").textContent =
      `${nodes.length}/8 model nodes · ${state.shared_training_characters.toLocaleString()} shared training characters.`;
  }
  renderAgentGraph();
}

async function refreshAgentGraph() {
  try {
    localAgentGraph = await localModelRequest("agents/graph");
    localAgentGraphError = null;
  } catch (error) {
    localAgentGraph = null;
    localAgentGraphError = error.message;
  }
  renderAgentGraph();
}

function renderAgentGraph() {
  const nodes = localAgentNetwork?.nodes ?? [];
  const trained = nodes.length > 0 && nodes.every((node) => node.trained_characters > 0);
  const active = Boolean(localAgentGraph?.episode_active);
  const awaitingReward = Boolean(localAgentGraph?.awaiting_reward);
  $("#agentGraphSummary").textContent = localAgentGraph
    ? `${localAgentGraph.active_edges} edges · ${localAgentGraph.episodes_trained} RL episodes`
    : "Graph unavailable";
  $("#trainAgentGraph").disabled = !trained || active;
  $("#startAgentGraphEpisode").disabled = !trained || active;
  $("#stepAgentGraph").disabled = !trained || !active || awaitingReward;
  $("#submitAgentGraphReward").disabled = !trained || !active || !awaitingReward;
  if (localAgentGraphError) {
    $("#agentGraphEpisodeStatus").textContent = `Graph status unavailable: ${localAgentGraphError}`;
  } else if (!nodes.length) {
    $("#agentGraphEpisodeStatus").textContent = "Create and train model nodes to use the latent graph.";
  } else if (!trained) {
    $("#agentGraphEpisodeStatus").textContent = "Train every model node before starting a graph episode.";
  } else if (awaitingReward) {
    $("#agentGraphEpisodeStatus").textContent = "Evaluate the selected action, then submit its reward.";
  } else if (active) {
    $("#agentGraphEpisodeStatus").textContent = "Episode active. Provide the next task-state prompt to select an action.";
  } else {
    $("#agentGraphEpisodeStatus").textContent = "Ready to start a graph-policy episode.";
  }
}

async function refreshReasoningTasks() {
  const picker = $("#reasoningTaskSelect");
  try {
    const result = await localModelRequest("agents/datasets");
    reasoningTaskCatalog = result.tasks;
    picker.replaceChildren();
    for (const [name, task] of Object.entries(reasoningTaskCatalog)) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = task.label;
      picker.append(option);
    }
    picker.disabled = false;
    $("#reasoningDatasetStatus").textContent =
      `${Object.keys(reasoningTaskCatalog).length} curated procedural datasets available from ` +
      `${result.source}.`;
  } catch (error) {
    reasoningTaskCatalog = null;
    picker.replaceChildren();
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "Reasoning Gym unavailable";
    picker.append(option);
    picker.disabled = true;
    $("#reasoningDatasetStatus").textContent = error.message.includes("pip install")
      ? "Install the optional dependency with: pip install -e '.[reasoning]'."
      : `Reasoning tasks unavailable: ${error.message}`;
  }
  renderAgentNetwork(localAgentNetwork);
}

async function trainLocalModel(source, text = undefined) {
  const epochs = Number($("#modelTrainingEpochs").value);
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
    throw new Error("Training epochs must be a whole number from 1 to 50.");
  }
  const button = source === "examples" ? $("#trainFromExamples") : $("#trainFromCorpus");
  button.disabled = true;
  $("#modelTrainingStatus").textContent = localModelHealth
    ? "Training the local character model…"
    : "Training the browser-local character model on this computer…";
  try {
    if (localModelHealth) {
      const result = await localModelRequest("train", {
        source,
        ...(text === undefined ? {} : { text }),
        epochs,
      });
      const loss = result.losses.at(-1);
      $("#modelTrainingStatus").textContent =
        `Training complete · ${result.training_characters.toLocaleString()} cumulative characters · ` +
        `final loss ${loss.toFixed(3)}. Generation is experimental; evaluate outputs carefully.`;
      await refreshLocalModel();
    } else {
      const trainingText =
        text ??
        examples.map((example) => `User: ${example.prompt}\nAssistant: ${example.response}`).join("\n");
      if (!trainingText) throw new Error("Teach at least one example or choose a training document first.");
      const result = localCharacterModel.train(trainingText, epochs);
      persistLocalCharacterModel();
      $("#localModelBadge").textContent = "Browser model trained";
      $("#localModelStatus").textContent =
        `Browser-local model trained on this computer · ${result.trainingCharacters.toLocaleString()} ` +
        "cumulative training characters. No separate server is needed.";
      $("#modelTrainingStatus").textContent =
        `Training complete · ${result.trainingCharacters.toLocaleString()} cumulative characters. ` +
        "This compact character model is experimental; evaluate outputs carefully.";
    }
  } finally {
    button.disabled =
      (localModelHealth
        ? false
        : source === "examples" && examples.length === 0) ||
      (source === "examples" &&
        localModelHealth?.conversation_examples === 0 &&
        examples.length === 0);
  }
}

function makeId() {
  return globalThis.crypto?.randomUUID?.() ?? `network-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function syncModelPicker() {
  const picker = $("#modelPicker");
  picker.replaceChildren();
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.name;
    picker.append(option);
  }
  picker.value = activeModelId;
}

function switchModel(id) {
  const selected = models.find((item) => item.id === id);
  if (!selected) return;
  saveWorkspace();
  activeModelId = selected.id;
  network = createNetworkFromJSON(selected.model);
  modelName = selected.name;
  customDataset = Array.isArray(selected.dataset) ? selected.dataset : null;
  if (customDataset) network.loss(customDataset);
  policy = {
    baseline: Number.isFinite(selected.policy?.baseline) ? selected.policy.baseline : 0,
    updates: Number.isInteger(selected.policy?.updates) ? selected.policy.updates : 0,
    pending: null,
  };
  render();
  saveWorkspace();
}

function createModel() {
  const index = models.length + 1;
  const name = `BRNA network ${index}`;
  activeModelId = makeId();
  modelName = name;
  network = new MessagePassingNetwork({}, { seed: index });
  customDataset = null;
  policy = { baseline: 0, updates: 0, pending: null };
  models.push({ id: activeModelId, name, model: network.toJSON(), policy: { baseline: 0, updates: 0 } });
  render();
  saveWorkspace();
  announce("New network created. Configure its layers before training.");
  setPage("design");
}

function setPage(name) {
  pages.forEach((page) => page.classList.toggle("active", page.dataset.pagePanel === name));
  navLinks.forEach((link) => {
    const active = link.dataset.page === name;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
}

function architecture() {
  return [...layerList.querySelectorAll("[data-layer-size]")].map((input) => {
    const size = Number(input.value);
    if (!Number.isInteger(size)) throw new Error("Layer sizes must be whole numbers.");
    return size;
  });
}

function renderLayerEditor(sizes = network.sizes) {
  layerList.replaceChildren();
  sizes.forEach((size, index) => {
    const row = document.createElement("div");
    row.className = "layer-row";
    const description = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = ["Input", "Hidden", "Output"][index === 0 ? 0 : index === sizes.length - 1 ? 2 : 1];
    const kind = document.createElement("span");
    kind.className = "layer-kind";
    kind.textContent = index === 0 ? "Feature vector" : index === sizes.length - 1 ? (sizes.at(-1) === 1 ? "Sigmoid output" : "Softmax actions") : "Tanh activation";
    description.append(title, document.createElement("br"), kind);
    const input = document.createElement("input");
    input.type = "number";
    input.min = "1";
    input.max = "32";
    input.step = "1";
    input.value = String(size);
    input.dataset.layerSize = "true";
    input.setAttribute("aria-label", `${title.textContent} layer neuron count`);
    row.append(description, input);
    if (index > 0 && index < sizes.length - 1) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "layer-remove";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Remove hidden layer ${index}`);
      remove.addEventListener("click", () => {
        const sizes = architecture();
        sizes.splice(index, 1);
        rebuild(sizes);
      });
      row.append(remove);
    } else {
      const spacer = document.createElement("span");
      spacer.setAttribute("aria-hidden", "true");
      row.append(spacer);
    }
    layerList.append(row);
  });
}

function rebuild(sizes) {
  const validated = validateArchitecture(sizes);
  network = new DenseNetwork(validated, {
    seed: Date.now() >>> 0,
    learningRate: Number($("#learningRate").value),
  });
  customDataset = null;
  policy = { baseline: 0, updates: 0, pending: null };
  render();
  saveWorkspace();
  announce("New architecture applied. Weights were reinitialized.");
}

function messageConfiguration() {
  const value = (selector) => Number($(selector).value);
  const communication = $("#graphCommunication").value;
  return {
    nodes: value("#graphNodes"),
    features: value("#graphFeatures"),
    hidden: value("#graphHidden"),
    hops: value("#graphHops"),
    readoutNode: value("#graphReadout"),
    indegree: value("#graphIndegree"),
    outputs: value("#graphOutputs"),
    communication,
  };
}

function rebuildMessageNetwork() {
  network = new MessagePassingNetwork(messageConfiguration(), {
    seed: Date.now() >>> 0,
    learningRate: Number($("#learningRate").value),
  });
  customDataset = null;
  policy = { baseline: 0, updates: 0, pending: null };
  render();
  saveWorkspace();
  announce("BRNA message-passing architecture applied. Weights were reinitialized.");
}

function getDataset() {
  if (customDataset) return customDataset;
  const inputCount = network.sizes[0];
  const outputCount = network.sizes.at(-1);
  if (network.kind === "dense" && inputCount === 2 && outputCount === 1) return DEFAULT_DATASET;
  const sampleCount = Math.max(4, Math.min(16, 2 ** Math.min(inputCount, 4)));
  return Array.from({ length: sampleCount }, (_, sampleIndex) => {
    const input = Array.from({ length: inputCount }, (_, feature) =>
      feature < 4 ? (sampleIndex >> feature) & 1 : 0,
    );
    const active = input.slice(0, Math.min(inputCount, 4)).reduce((sum, value) => sum + value, 0);
    if (outputCount === 1) return { input, target: [active % 2] };
    const classIndex = active % outputCount;
    return {
      input,
      target: Array.from({ length: outputCount }, (_, index) => Number(index === classIndex)),
    };
  });
}

function renderDataset() {
  const rows = $("#datasetRows");
  rows.replaceChildren();
  $("#datasetEditor").value = JSON.stringify(getDataset(), null, 2);
  for (const example of getDataset()) {
    const row = document.createElement("tr");
    const input = document.createElement("td");
    const target = document.createElement("td");
    input.textContent = JSON.stringify(example.input);
    target.textContent = JSON.stringify(example.target);
    row.append(input, target);
    rows.append(row);
  }
}

function renderDiagram() {
  const host = $("#networkDiagram");
  if (network.kind === "message-passing") {
    const { nodes, features, hidden, hops, readoutNode, outputs, communication } = network.config;
    const width = 440;
    const height = 270;
    const center = { x: width / 2, y: height / 2 - 8 };
    const radius = Math.min(104, 35 + nodes * 2.4);
    const positions = Array.from({ length: nodes }, (_, index) => {
      const angle = (2 * Math.PI * index) / nodes - Math.PI / 2;
      return { x: center.x + Math.cos(angle) * radius, y: center.y + Math.sin(angle) * radius };
    });
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("width", "100%");
    svg.setAttribute("height", String(height));
    svg.setAttribute("role", "presentation");
    for (let receiver = 0; receiver < nodes; receiver += 1) {
      for (let sender = 0; sender < nodes; sender += 1) {
        if (!network.adjacency[receiver][sender]) continue;
        const line = document.createElementNS(svg.namespaceURI, "line");
        line.setAttribute("x1", String(positions[sender].x));
        line.setAttribute("y1", String(positions[sender].y));
        line.setAttribute("x2", String(positions[receiver].x));
        line.setAttribute("y2", String(positions[receiver].y));
        line.setAttribute("stroke", "#385744");
        line.setAttribute("stroke-width", "1");
        svg.append(line);
      }
    }
    positions.forEach(({ x, y }, index) => {
      const circle = document.createElementNS(svg.namespaceURI, "circle");
      circle.setAttribute("cx", String(x));
      circle.setAttribute("cy", String(y));
      circle.setAttribute("r", "12");
      circle.setAttribute("fill", index === readoutNode ? "#1c3040" : "#1d3728");
      circle.setAttribute("stroke", index === readoutNode ? "#75a9c7" : "#71ad83");
      circle.setAttribute("stroke-width", "2");
      svg.append(circle);
      const label = document.createElementNS(svg.namespaceURI, "text");
      label.setAttribute("x", String(x));
      label.setAttribute("y", String(y + 4));
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("fill", "#e4ece5");
      label.setAttribute("font-size", "9");
      label.textContent = String(index);
      svg.append(label);
    });
    const caption = document.createElementNS(svg.namespaceURI, "text");
    caption.setAttribute("x", String(center.x));
    caption.setAttribute("y", String(height - 8));
    caption.setAttribute("text-anchor", "middle");
    caption.setAttribute("fill", "#9eafa3");
    caption.setAttribute("font-size", "11");
    caption.textContent = `${nodes} nodes · ${network.adjacency.flat().reduce((sum, value) => sum + value, 0)} directed edges`;
    svg.append(caption);
    host.replaceChildren(svg);
    $("#architectureSummary").replaceChildren(
      ...[
        `${features} features / node`,
        `${hidden} hidden units`,
        `${hops} message-passing hops`,
        `Readout node ${readoutNode}`,
        `${outputs} output${outputs === 1 ? "" : "s"}`,
        `${communication} communication`,
      ].map((text) => {
        const chip = document.createElement("span");
        chip.textContent = text;
        return chip;
      }),
    );
    $("#parameterCount").textContent = `${network.parameterCount().toLocaleString()} parameters`;
    return;
  }
  const sizes = network.sizes;
  const width = Math.max(400, sizes.length * 150);
  const height = 220;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", String(height));
  svg.setAttribute("role", "presentation");
  const positions = sizes.map((size, layerIndex) => {
    const x = 40 + (width - 80) * (layerIndex / (sizes.length - 1));
    const shown = Math.min(size, 7);
    const spacing = Math.min(30, 150 / Math.max(1, shown - 1));
    return Array.from({ length: shown }, (_, nodeIndex) => ({
      x,
      y: height / 2 + (nodeIndex - (shown - 1) / 2) * spacing,
    }));
  });
  for (let layer = 0; layer < positions.length - 1; layer += 1) {
    for (const from of positions[layer]) {
      for (const to of positions[layer + 1]) {
        const line = document.createElementNS(svg.namespaceURI, "line");
        line.setAttribute("x1", String(from.x + 12));
        line.setAttribute("y1", String(from.y));
        line.setAttribute("x2", String(to.x - 12));
        line.setAttribute("y2", String(to.y));
        line.setAttribute("stroke", "#385744");
        line.setAttribute("stroke-width", "1");
        svg.append(line);
      }
    }
  }
  positions.forEach((layerNodes, layerIndex) => {
    layerNodes.forEach(({ x, y }, nodeIndex) => {
      const circle = document.createElementNS(svg.namespaceURI, "circle");
      circle.setAttribute("cx", String(x));
      circle.setAttribute("cy", String(y));
      circle.setAttribute("r", "12");
      circle.setAttribute("fill", layerIndex === sizes.length - 1 ? "#1c3040" : "#1d3728");
      circle.setAttribute("stroke", layerIndex === sizes.length - 1 ? "#75a9c7" : "#71ad83");
      circle.setAttribute("stroke-width", "2");
      svg.append(circle);
      if (sizes[layerIndex] > layerNodes.length && nodeIndex === layerNodes.length - 1) {
        const dots = document.createElementNS(svg.namespaceURI, "text");
        dots.setAttribute("x", String(x));
        dots.setAttribute("y", String(y + 29));
        dots.setAttribute("text-anchor", "middle");
        dots.setAttribute("fill", "#9eafa3");
        dots.setAttribute("font-size", "11");
        dots.textContent = `+${sizes[layerIndex] - layerNodes.length} neurons`;
        svg.append(dots);
      }
    });
    const label = document.createElementNS(svg.namespaceURI, "text");
    label.setAttribute("x", String(positions[layerIndex][0].x));
    label.setAttribute("y", String(height - 8));
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("fill", "#9eafa3");
    label.setAttribute("font-size", "11");
    label.textContent = `${["Input", ...Array(sizes.length - 2).fill("Hidden"), "Output"][layerIndex]} · ${sizes[layerIndex]}`;
    svg.append(label);
  });
  host.replaceChildren(svg);
  $("#architectureSummary").replaceChildren(
    ...sizes.map((size, index) => {
      const chip = document.createElement("span");
      chip.textContent = `${index === 0 ? "Input" : index === sizes.length - 1 ? "Output" : `Hidden ${index}`} ${size}`;
      return chip;
    }),
  );
  $("#parameterCount").textContent = `${network.parameterCount().toLocaleString()} parameters`;
}

function renderEvaluation() {
  const host = $("#evaluationResults");
  const dataset = getDataset();
  const loss = network.loss(dataset);
  const accuracy = evaluateBinaryAccuracy(network, dataset);
  $("#lossSummary").textContent = loss.toFixed(4);
  $("#fitnessSummary").textContent = accuracy === null ? "Multi-class" : `${(accuracy * 100).toFixed(0)}% accuracy`;
  const table = document.createElement("table");
  const header = document.createElement("thead");
  header.innerHTML = "<tr><th>Input</th><th>Target</th><th>Network output</th></tr>";
  const body = document.createElement("tbody");
  dataset.forEach((example) => {
    const row = document.createElement("tr");
    for (const value of [
      JSON.stringify(example.input),
      JSON.stringify(example.target),
      JSON.stringify(network.forward(example.input).map((item) => Number(item.toFixed(4)))),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    body.append(row);
  });
  table.append(header, body);
  const summary = document.createElement("p");
  summary.className = "status-message";
  summary.textContent = `Mean cross-entropy loss: ${loss.toFixed(5)}${accuracy === null ? "" : ` · Accuracy: ${(accuracy * 100).toFixed(1)}%`}`;
  host.replaceChildren(summary, table);
}

function renderConversations() {
  $("#exampleCount").textContent = `${examples.length} example${examples.length === 1 ? "" : "s"}`;
  const host = $("#exampleList");
  host.replaceChildren();
  if (examples.length === 0) {
    const empty = document.createElement("p");
    empty.className = "status-message";
    empty.textContent = "No examples yet. Teach a response above to start a local conversation memory.";
    host.append(empty);
    return;
  }
  examples.forEach((example, index) => {
    const item = document.createElement("div");
    item.className = "example-item";
    for (const [label, content] of [["Prompt", example.prompt], ["Taught response", example.response]]) {
      const column = document.createElement("div");
      const heading = document.createElement("strong");
      heading.textContent = label;
      const paragraph = document.createElement("p");
      paragraph.textContent = content;
      column.append(heading, paragraph);
      item.append(column);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "example-delete";
    remove.textContent = "Remove";
    remove.addEventListener("click", () => {
      examples.splice(index, 1);
      saveWorkspace();
      renderConversations();
    });
    item.append(remove);
    host.append(item);
  });
}

function renderPolicy() {
  const outputs = network.sizes.at(-1);
  $("#policyFeatureHint").textContent = `${network.sizes[0]} values`;
  $("#actionCount").textContent = `${outputs} action${outputs === 1 ? "" : "s"}`;
  $("#rewardBaseline").textContent = policy.baseline.toFixed(2);
  $("#submitReward").disabled = !policy.pending;
  const host = $("#policyOutputs");
  host.replaceChildren();
  for (let index = 0; index < outputs; index += 1) {
    const row = document.createElement("div");
    row.className = "policy-bar";
    const label = document.createElement("span");
    label.textContent = `Action ${index + 1}`;
    const track = document.createElement("div");
    track.className = "policy-bar-track";
    const fill = document.createElement("div");
    fill.className = "policy-bar-fill";
    const value = document.createElement("span");
    value.className = "policy-bar-value";
    value.textContent = "—";
    track.append(fill);
    row.append(label, track, value);
    host.append(row);
  }
}

function render() {
  $("#modelName").value = modelName;
  $("#activeModelTitle").textContent = modelName;
  syncModelPicker();
  $("#inputSummary").textContent = `${network.sizes[0]} feature${network.sizes[0] === 1 ? "" : "s"}`;
  $("#outputSummary").textContent = `${network.sizes.at(-1)} output${network.sizes.at(-1) === 1 ? "" : "s"}`;
  $("#learningRate").value = String(network.learningRate);
  const isMessagePassing = network.kind === "message-passing";
  $("#networkFamily").value = network.kind;
  $("#denseControls").hidden = isMessagePassing;
  $("#messageControls").hidden = !isMessagePassing;
  if (isMessagePassing) {
    const config = network.config;
    $("#graphNodes").value = String(config.nodes);
    $("#graphFeatures").value = String(config.features);
    $("#graphHidden").value = String(config.hidden);
    $("#graphHops").value = String(config.hops);
    $("#graphReadout").value = String(config.readoutNode);
    $("#graphIndegree").value = String(config.indegree);
    $("#graphOutputs").value = String(config.outputs);
    $("#graphCommunication").value = config.communication;
  }
  renderLayerEditor(isMessagePassing ? [2, 4, 1] : network.sizes);
  renderDiagram();
  renderDataset();
  renderEvaluation();
  renderConversations();
  renderPolicy();
}

function announce(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
}

function parseFeatures() {
  const values = $("#policyFeatures").value.split(",").map((value) => Number(value.trim()));
  if (
    values.length !== network.sizes[0] ||
    values.some((value, index) => !$("#policyFeatures").value.split(",")[index].trim() || !Number.isFinite(value))
  ) {
    throw new Error(`Enter exactly ${network.sizes[0]} finite comma-separated feature values.`);
  }
  return values;
}

function renderActionProbabilities(probabilities) {
  $$("#policyOutputs .policy-bar").forEach((row, index) => {
    const probability = probabilities[index] ?? 0;
    row.querySelector(".policy-bar-fill").style.width = `${Math.max(0, Math.min(1, probability)) * 100}%`;
    row.querySelector(".policy-bar-value").textContent = `${(probability * 100).toFixed(1)}%`;
  });
}

function downloadModel() {
  const data = {
    format: "brna-studio-network",
    name: modelName,
    created: new Date().toISOString(),
    model: network.toJSON(),
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${modelName.trim().replace(/[^a-z0-9_-]+/gi, "-") || "brna-network"}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

navLinks.forEach((link) => link.addEventListener("click", (event) => {
  event.preventDefault();
  setPage(link.dataset.page);
}));
window.addEventListener("hashchange", () => {
  const target = location.hash.slice(1);
  if (pages.some((page) => page.dataset.pagePanel === target)) setPage(target);
});

$("#modelName").addEventListener("change", (event) => {
  modelName = event.target.value.trim() || "BRNA prototype";
  render();
  saveWorkspace();
});
$("#modelPicker").addEventListener("change", (event) => switchModel(event.target.value));
$("#newModel").addEventListener("click", createModel);
$("#networkFamily").addEventListener("change", (event) => {
  const isMessagePassing = event.target.value === "message-passing";
  $("#denseControls").hidden = isMessagePassing;
  $("#messageControls").hidden = !isMessagePassing;
  if (!isMessagePassing) {
    renderLayerEditor(network.kind === "dense" ? network.sizes : [2, 4, 1]);
  }
});
$("#addLayer").addEventListener("click", () => {
  const sizes = architecture();
  if (sizes.length >= 8) return announce("A network can have up to 6 hidden layers.");
  sizes.splice(-1, 0, 4);
  rebuild(sizes);
});
$("#buildNetwork").addEventListener("click", () => {
  try {
    rebuild(architecture());
  } catch (error) {
    announce(error.message);
  }
});
$("#buildMessageNetwork").addEventListener("click", () => {
  try {
    rebuildMessageNetwork();
  } catch (error) {
    announce(`BRNA architecture was not applied: ${error.message}`);
  }
});
$("#applyDataset").addEventListener("click", () => {
  try {
    const parsed = JSON.parse($("#datasetEditor").value);
    if (!Array.isArray(parsed) || parsed.length > 500) {
      throw new Error("Dataset must be a JSON array with no more than 500 examples.");
    }
    network.loss(parsed);
    customDataset = parsed.map((example) => ({
      input: example.input.slice(),
      target: example.target.slice(),
    }));
    renderDataset();
    renderEvaluation();
    saveWorkspace();
    announce(`Applied ${customDataset.length} training examples.`);
  } catch (error) {
    announce(`Dataset was not applied: ${error.message}`);
  }
});
$("#useDemoDataset").addEventListener("click", () => {
  customDataset = null;
  renderDataset();
  renderEvaluation();
  saveWorkspace();
  announce("Demo examples restored.");
});
$("#exportModel").addEventListener("click", downloadModel);
$("#exportPolicy").addEventListener("click", downloadModel);
$("#importModel").addEventListener("change", async (event) => {
  const [file] = event.target.files ?? [];
  if (!file) return;
  try {
    const parsed = JSON.parse(await file.text());
    const modelData = parsed.format === "brna-studio-network" ? parsed.model : parsed;
    network = createNetworkFromJSON(modelData);
    customDataset = null;
    if (typeof parsed.name === "string" && parsed.name.trim()) modelName = parsed.name;
    policy = { baseline: 0, updates: 0, pending: null };
    render();
    saveWorkspace();
    announce("Network imported.");
  } catch (error) {
    announce(`Import failed: ${error.message}`);
  } finally {
    event.target.value = "";
  }
});

$("#learningRate").addEventListener("change", (event) => {
  const value = Number(event.target.value);
  if (!Number.isFinite(value) || value <= 0 || value > 1) {
    announce("Learning rate must be greater than 0 and no more than 1.");
    return;
  }
  network.learningRate = value;
  saveWorkspace();
});

$("#trainSupervised").addEventListener("click", () => {
  const button = $("#trainSupervised");
  try {
    const learningRate = Number($("#learningRate").value);
    const epochs = Number($("#epochCount").value);
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) throw new Error("Learning rate must be greater than 0 and no more than 1.");
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 1000) throw new Error("Epochs must be a whole number from 1 to 1000.");
    const dataset = getDataset();
    button.disabled = true;
    network.learningRate = learningRate;
    let epoch = 0;
    const trainChunk = () => {
      try {
        const limit = Math.min(epochs, epoch + 10);
        for (; epoch < limit; epoch += 1) network.trainEpoch(dataset, learningRate);
        $("#supervisedStatus").textContent = `Training · epoch ${epoch}/${epochs} · loss ${network.loss(dataset).toFixed(5)}`;
        if (epoch < epochs) requestAnimationFrame(trainChunk);
        else {
          $("#supervisedStatus").textContent = `Training complete · loss ${network.loss(dataset).toFixed(5)}`;
          button.disabled = false;
          renderEvaluation();
          saveWorkspace();
        }
      } catch (error) {
        button.disabled = false;
        announce(error.message);
      }
    };
    trainChunk();
  } catch (error) {
    announce(error.message);
  }
});

$("#trainEvolution").addEventListener("click", () => {
  const button = $("#trainEvolution");
  try {
    const size = Number($("#populationSize").value);
    const generations = Number($("#generationCount").value);
    const mutationScale = Number($("#mutationScale").value);
    const seed = Number($("#evolutionSeed").value);
    if (!Number.isInteger(generations) || generations < 1 || generations > 500) throw new Error("Generations must be a whole number from 1 to 500.");
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) throw new Error("Seed must be an unsigned 32-bit integer.");
    if (!Number.isFinite(mutationScale) || mutationScale <= 0 || mutationScale > 2) throw new Error("Mutation scale must be greater than 0 and no more than 2.");
    let population = initialPopulation(network, size, { seed, mutationScale });
    const dataset = getDataset();
    let generation = 0;
    let bestFitness = -network.loss(dataset);
    button.disabled = true;
    const evolveChunk = () => {
      try {
        const limit = Math.min(generations, generation + 5);
        for (; generation < limit; generation += 1) {
          const result = evolveGeneration(network, dataset, population, {
            seed: (seed + generation) >>> 0,
            mutationScale,
          });
          population = result.population;
          if (result.bestFitness > bestFitness) {
            bestFitness = result.bestFitness;
            network.setGenome(result.bestGenome);
          }
        }
        $("#evolutionStatus").textContent = `Generation ${generation}/${generations} · best fitness ${bestFitness.toFixed(5)} · loss ${(-bestFitness).toFixed(5)}`;
        renderEvaluation();
        if (generation < generations) requestAnimationFrame(evolveChunk);
        else {
          button.disabled = false;
          saveWorkspace();
        }
      } catch (error) {
        button.disabled = false;
        announce(error.message);
      }
    };
    evolveChunk();
  } catch (error) {
    announce(error.message);
  }
});
$("#evaluateNetwork").addEventListener("click", renderEvaluation);

$("#saveExample").addEventListener("click", async () => {
  const prompt = $("#teachPrompt").value.trim();
  const response = $("#teachResponse").value.trim();
  if (!prompt || !response) {
    $("#teachStatus").textContent = "Both a prompt and a taught response are required.";
    return;
  }
  if (!examples.some((item) => item.prompt === prompt && item.response === response)) {
    examples.unshift({ prompt, response });
  }
  $("#teachPrompt").value = "";
  $("#teachResponse").value = "";
  $("#teachStatus").textContent = "Example added to browser conversation memory.";
  saveWorkspace();
  renderConversations();
  if (localModelHealth) {
    try {
      await localModelRequest("teach", { prompt, response });
      $("#teachStatus").textContent = "Example saved in the browser and synced to the self-hosted model server.";
    } catch (error) {
      $("#teachStatus").textContent = `Saved in this browser, but could not sync to the model server: ${error.message}`;
    }
  }
});
$("#askMemory").addEventListener("click", async () => {
  const prompt = $("#chatPrompt").value.trim();
  const response = $("#chatResponse");
  if (!prompt) {
    response.textContent = "Enter a prompt to search your taught examples.";
    response.classList.remove("has-answer");
    return;
  }
  if (localModelHealth) {
    response.textContent = "Asking the self-hosted model…";
    try {
      const result = await localModelRequest("chat", { prompt });
      const source = result.source === "local-model" ? "Generated by your local model" : "Retrieved from a taught example";
      response.textContent = `${result.response}\n\n${source}`;
      response.classList.add("has-answer");
    } catch (error) {
      response.textContent = `The local model could not answer: ${error.message}`;
      response.classList.remove("has-answer");
    }
    return;
  }
  const match = retrieveReply(examples, prompt);
  if (match) {
    response.textContent = `${match.response}\n\nRetrieved from your examples · similarity ${(match.score * 100).toFixed(0)}%`;
    response.classList.add("has-answer");
  } else if (localCharacterModel.trained) {
    try {
      const generated = localCharacterModel.generate(`User: ${prompt}\nAssistant:`);
      if (!generated) throw new Error("The model generated an empty response; train it with more text.");
      response.textContent = `${generated}\n\nGenerated by the browser-local character model`;
      response.classList.add("has-answer");
    } catch (error) {
      response.textContent = `The browser-local model could not answer: ${error.message}`;
      response.classList.remove("has-answer");
    }
  } else {
    response.textContent =
      "No sufficiently similar taught response was found. Train the browser-local model or teach a matching example first.";
    response.classList.remove("has-answer");
  }
});
$("#refreshLocalModel").addEventListener("click", refreshLocalModel);

$("#createAgentNode").addEventListener("click", async () => {
  try {
    const result = await localModelRequest("agents/create", {
      name: $("#agentName").value,
      role: $("#agentRole").value,
      hidden_size: Number($("#agentHiddenSize").value),
    });
    $("#agentBuilderStatus").textContent = `Created ${result.name}; model parameters are independent.`;
    await refreshLocalModel();
  } catch (error) {
    $("#agentBuilderStatus").textContent = `Could not create agent: ${error.message}`;
  }
});

$("#trainReasoningDataset").addEventListener("click", async () => {
  try {
    const taskName = $("#reasoningTaskSelect").value;
    const size = Number($("#reasoningExampleCount").value);
    const seed = Number($("#reasoningSeed").value);
    const epochs = Number($("#agentTrainingEpochs").value);
    if (!taskName) throw new Error("Choose an available reasoning task.");
    if (!Number.isInteger(size) || size < 1 || size > 128) {
      throw new Error("Generated examples must be a whole number from 1 to 128.");
    }
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) {
      throw new Error("Dataset seed must be a whole number from 0 to 4294967295.");
    }
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
      throw new Error("Epochs must be a whole number from 1 to 50.");
    }
    $("#trainReasoningDataset").disabled = true;
    $("#reasoningDatasetStatus").textContent = "Generating verified tasks and training every model node…";
    const result = await localModelRequest("agents/datasets/train", {
      task_name: taskName,
      size,
      seed,
      epochs,
    });
    $("#reasoningDatasetStatus").textContent =
      `Trained ${result.trained_nodes} nodes on ${result.examples_generated} generated ` +
      `${result.dataset} examples · seed ${result.seed} · shared corpus ` +
      `${result.training_characters.toLocaleString()} characters.`;
    await refreshLocalModel();
  } catch (error) {
    $("#reasoningDatasetStatus").textContent = `Dataset training failed: ${error.message}`;
  } finally {
    renderAgentNetwork(localAgentNetwork);
  }
});

$("#runVerifiedEpisode").addEventListener("click", async () => {
  try {
    const taskName = $("#reasoningTaskSelect").value;
    const seed = Number($("#reasoningSeed").value);
    const index = Number($("#reasoningEpisodeIndex").value);
    if (!taskName) throw new Error("Choose an available reasoning task.");
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) {
      throw new Error("Dataset seed must be a whole number from 0 to 4294967295.");
    }
    if (!Number.isInteger(index) || index < 0 || index > 127) {
      throw new Error("Task index must be a whole number from 0 to 127.");
    }
    $("#runVerifiedEpisode").disabled = true;
    $("#verifiedEpisodeOutput").textContent = "Generating, solving, verifying, and applying reward…";
    const result = await localModelRequest("agents/reasoning-episode", {
      task_name: taskName,
      seed,
      index,
      max_new_tokens: 48,
    });
    $("#verifiedEpisodeOutput").textContent =
      `Task: ${result.question}\n` +
      `Model response: ${result.model_response}\n` +
      `Expected answer: ${result.expected_answer}\n` +
      `Verifier score: ${result.verifier_score.toFixed(3)} · RL reward: ${result.reward.toFixed(3)}\n` +
      `Learning update: ${result.learning.algorithm}; ${result.learning.updated_nodes} nodes, ` +
      `${result.learning.trajectory_steps} sampled outputs.`;
    await refreshLocalModel();
  } catch (error) {
    $("#verifiedEpisodeOutput").textContent = `Verified episode failed: ${error.message}`;
  } finally {
    renderAgentNetwork(localAgentNetwork);
  }
});

$("#runGRPOEpisode").addEventListener("click", async () => {
  try {
    const taskName = $("#reasoningTaskSelect").value;
    const seed = Number($("#reasoningSeed").value);
    const index = Number($("#reasoningEpisodeIndex").value);
    const group_size = Number($("#grpoGroupSize").value);
    const kl_beta = Number($("#grpoKLBeta").value);
    if (!taskName) throw new Error("Choose an available reasoning task.");
    if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) {
      throw new Error("Dataset seed must be a whole number from 0 to 4294967295.");
    }
    if (!Number.isInteger(index) || index < 0 || index > 127) {
      throw new Error("Task index must be a whole number from 0 to 127.");
    }
    if (!Number.isInteger(group_size) || group_size < 2 || group_size > 8) {
      throw new Error("GRPO group size must be a whole number from 2 to 8.");
    }
    if (!Number.isFinite(kl_beta) || kl_beta < 0 || kl_beta > 1) {
      throw new Error("Frozen-base KL weight must be between 0 and 1.");
    }
    $("#runGRPOEpisode").disabled = true;
    $("#grpoEpisodeStatus").textContent = "Sampling candidate groups, verifying answers, and optimizing adapters…";
    const result = await localModelRequest("agents/grpo-episode", {
      task_name: taskName,
      seed,
      index,
      group_size,
      kl_beta,
    });
    $("#grpoEpisodeStatus").textContent =
      `${result.algorithm} · ${result.verified_candidates} verified candidates · ` +
      `mean reward ${result.mean_reward.toFixed(3)} · mean KL ${result.mean_kl.toFixed(4)}.`;
    await refreshLocalModel();
  } catch (error) {
    $("#grpoEpisodeStatus").textContent = `GRPO episode failed: ${error.message}`;
  } finally {
    renderAgentNetwork(localAgentNetwork);
  }
});

$("#trainAgentNetwork").addEventListener("click", async () => {
  try {
    const text = $("#agentTrainingText").value;
    if (!text.trim()) throw new Error("Enter a shared training corpus first.");
    const epochs = Number($("#agentTrainingEpochs").value);
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
      throw new Error("Epochs must be a whole number from 1 to 50.");
    }
    $("#trainAgentNetwork").disabled = true;
    const result = await localModelRequest("agents/train", { text, epochs });
    $("#agentTrainingStatus").textContent =
      `Training complete · ${result.trained_nodes} model nodes · ` +
      `${result.training_characters.toLocaleString()} shared characters.`;
    await refreshLocalModel();
  } catch (error) {
    $("#agentTrainingStatus").textContent = `Agent training failed: ${error.message}`;
  } finally {
    renderAgentNetwork(localAgentNetwork);
  }
});

$("#uploadAgentCorpus").addEventListener("click", async () => {
  try {
    const [file] = $("#agentTrainingFile").files ?? [];
    if (!file) throw new Error("Choose a training file first.");
    if (file.size > 20_000_000) throw new Error("Training documents must not exceed 20 MB.");
    const epochs = Number($("#agentTrainingEpochs").value);
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
      throw new Error("Epochs must be a whole number from 1 to 50.");
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    let binary = "";
    for (let offset = 0; offset < bytes.length; offset += 0x8000) {
      binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
    }
    $("#agentTrainingStatus").textContent = `Extracting and training every model from ${file.name}…`;
    $("#uploadAgentCorpus").disabled = true;
    const result = await localModelRequest("agents/upload", {
      filename: file.name,
      content_base64: btoa(binary),
      epochs,
    });
    $("#agentTrainingStatus").textContent =
      `Trained ${result.trained_nodes} nodes from ${result.filename} · ` +
      `${result.extracted_characters.toLocaleString()} extracted characters.`;
    await refreshLocalModel();
  } catch (error) {
    $("#agentTrainingStatus").textContent = `Document training failed: ${error.message}`;
  } finally {
    renderAgentNetwork(localAgentNetwork);
  }
});

$("#runAgentNetwork").addEventListener("click", async () => {
  const prompt = $("#agentTaskPrompt").value.trim();
  if (!prompt) {
    $("#agentReinforcementStatus").textContent = "Enter a task prompt first.";
    return;
  }
  $("#runAgentNetwork").disabled = true;
  $("#agentReinforcementStatus").textContent = "Running private and shared model passes…";
  try {
    const result = await localModelRequest("agents/think", { prompt, max_new_tokens: 48 });
    const output = $("#agentCollectiveOutput");
    output.replaceChildren();
    for (const message of result.messages) {
      const item = document.createElement("div");
      item.className = "example-item";
      const details = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = `${message.name} · ${message.round} pass`;
      const text = document.createElement("p");
      text.textContent = message.text;
      details.append(title, text);
      item.append(details);
      output.append(item);
    }
    $("#agentReinforcementStatus").textContent =
      `Collective response: ${result.response} · rate this full episode from -1 to 1.`;
    $("#reinforceAgentNetwork").disabled = false;
  } catch (error) {
    $("#agentReinforcementStatus").textContent = `Collective run failed: ${error.message}`;
    $("#reinforceAgentNetwork").disabled = true;
  } finally {
    renderAgentNetwork(localAgentNetwork);
  }
});

$("#reinforceAgentNetwork").addEventListener("click", async () => {
  try {
    const reward = Number($("#agentEpisodeReward").value);
    if (!Number.isFinite(reward) || reward < -1 || reward > 1) {
      throw new Error("Episode reward must be between -1 and 1.");
    }
    const result = await localModelRequest("agents/feedback", { reward, discount: 0.95 });
    $("#agentReinforcementStatus").textContent =
      `Updated ${result.updated_nodes} models over ${result.trajectory_steps} turns · ` +
      `reward baseline ${result.baseline.toFixed(3)} · ${result.algorithm}.`;
    $("#reinforceAgentNetwork").disabled = true;
    await refreshLocalModel();
  } catch (error) {
    $("#agentReinforcementStatus").textContent = `Reinforcement update failed: ${error.message}`;
  }
});

$("#trainAgentGraph").addEventListener("click", async () => {
  try {
    const examples = JSON.parse($("#agentGraphExamples").value);
    if (!Array.isArray(examples) || examples.length === 0) {
      throw new Error("Provide a non-empty JSON array of labeled examples.");
    }
    const epochs = Number($("#agentGraphEpochs").value);
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 100) {
      throw new Error("Graph training epochs must be a whole number from 1 to 100.");
    }
    $("#trainAgentGraph").disabled = true;
    $("#agentGraphTrainingStatus").textContent = "Training graph transforms and directed edge gates…";
    const result = await localModelRequest("agents/graph/train", { examples, epochs });
    $("#agentGraphTrainingStatus").textContent =
      `Trained ${result.examples_trained} examples for ${result.updates} updates · ` +
      `mean gate ${result.mean_gate.toFixed(3)}.`;
    await refreshAgentGraph();
  } catch (error) {
    $("#agentGraphTrainingStatus").textContent = `Graph training failed: ${error.message}`;
  } finally {
    renderAgentGraph();
  }
});

$("#startAgentGraphEpisode").addEventListener("click", async () => {
  try {
    await localModelRequest("agents/graph/episode/start", {});
    await refreshAgentGraph();
    $("#agentGraphStepOutput").replaceChildren();
    $("#agentGraphEpisodeStatus").textContent = "Episode started. Enter a task-state prompt and request an action.";
  } catch (error) {
    $("#agentGraphEpisodeStatus").textContent = `Could not start graph episode: ${error.message}`;
  } finally {
    renderAgentGraph();
  }
});

$("#stepAgentGraph").addEventListener("click", async () => {
  try {
    const prompt = $("#agentGraphPrompt").value.trim();
    if (!prompt) throw new Error("Enter a task-state prompt first.");
    const max_new_tokens = Number($("#agentGraphTokens").value);
    if (!Number.isInteger(max_new_tokens) || max_new_tokens < 1 || max_new_tokens > 128) {
      throw new Error("Response tokens must be a whole number from 1 to 128.");
    }
    const result = await localModelRequest("agents/graph/step", { prompt, max_new_tokens });
    const output = $("#agentGraphStepOutput");
    output.replaceChildren();
    for (const message of result.messages) {
      const item = document.createElement("div");
      item.className = "example-item";
      const details = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = `${message.name} · ${message.role}`;
      const text = document.createElement("p");
      text.textContent = message.text;
      details.append(title, text);
      item.append(details);
      output.append(item);
    }
    const action = document.createElement("p");
    action.className = "field-help";
    action.textContent =
      `Selected action ${result.action.action} · probabilities: ` +
      result.action.probabilities.map((value) => value.toFixed(3)).join(", ");
    output.append(action);
    await refreshAgentGraph();
  } catch (error) {
    $("#agentGraphEpisodeStatus").textContent = `Graph action failed: ${error.message}`;
  } finally {
    renderAgentGraph();
  }
});

$("#submitAgentGraphReward").addEventListener("click", async () => {
  try {
    const reward = Number($("#agentGraphReward").value);
    if (!Number.isFinite(reward) || reward < -1 || reward > 1) {
      throw new Error("Observed reward must be between -1 and 1.");
    }
    const done = $("#agentGraphDone").value === "true";
    const result = await localModelRequest("agents/graph/reward", { reward, done });
    if (result.episode_complete) {
      $("#agentGraphEpisodeStatus").textContent =
        `Episode learned · ${result.updated_steps} actions · ` +
        `baseline ${result.baseline.toFixed(3)} · ${result.algorithm}.`;
      await refreshLocalModel();
    } else {
      $("#agentGraphEpisodeStatus").textContent = "Reward recorded. Continue the episode with the next task-state prompt.";
      await refreshAgentGraph();
    }
  } catch (error) {
    $("#agentGraphEpisodeStatus").textContent = `Reward update failed: ${error.message}`;
  } finally {
    renderAgentGraph();
  }
});

$("#trainFromExamples").addEventListener("click", async () => {
  try {
    await trainLocalModel("examples");
  } catch (error) {
    $("#modelTrainingStatus").textContent = `Model training failed: ${error.message}`;
  }
});
$("#trainFromCorpus").addEventListener("click", async () => {
  try {
    const [file] = $("#modelCorpusFile").files ?? [];
    if (!file) throw new Error("Choose a training document first.");
    if (file.size > 20_000_000) throw new Error("Training documents must not exceed 20 MB.");
    const extension = file.name.split(".").pop()?.toLowerCase();
    if (["txt", "md", "markdown"].includes(extension)) {
      const text = await file.text();
      await trainLocalModel("corpus", text);
    } else if (["pdf", "docx"].includes(extension)) {
      if (!localModelHealth) {
        throw new Error("PDF and DOCX extraction requires the optional local Python server; use .txt or .md without it.");
      }
      const epochs = Number($("#modelTrainingEpochs").value);
      if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
        throw new Error("Training epochs must be a whole number from 1 to 50.");
      }
      $("#modelTrainingStatus").textContent = "Extracting and training from the uploaded document…";
      const bytes = new Uint8Array(await file.arrayBuffer());
      let binary = "";
      for (let offset = 0; offset < bytes.length; offset += 0x8000) {
        binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
      }
      const result = await localModelRequest("upload", {
        filename: file.name,
        content_base64: btoa(binary),
        epochs,
      });
      const loss = result.losses.at(-1);
      $("#modelTrainingStatus").textContent =
        `Uploaded ${result.filename} (${result.file_type}) · extracted ${result.extracted_characters.toLocaleString()} characters · ` +
        `final loss ${loss.toFixed(3)}.`;
      await refreshLocalModel();
    } else {
      throw new Error("Use a .txt, .md, .docx, or .pdf training document.");
    }
  } catch (error) {
    $("#modelTrainingStatus").textContent = `Model training failed: ${error.message}`;
  }
});
$("#clearExamples").addEventListener("click", () => {
  if (examples.length === 0) return;
  examples = [];
  saveWorkspace();
  renderConversations();
  $("#chatResponse").textContent = "Responses come only from examples you have taught.";
  $("#chatResponse").classList.remove("has-answer");
});

$("#choosePolicyAction").addEventListener("click", () => {
  try {
    if (network.sizes.at(-1) < 2) throw new Error("Add at least two output neurons to use policy learning.");
    const input = parseFeatures();
    const probabilities = network.forward(input);
    const draw = Math.random();
    let cumulative = 0;
    let actionIndex = probabilities.length - 1;
    for (let index = 0; index < probabilities.length; index += 1) {
      cumulative += probabilities[index];
      if (draw <= cumulative) {
        actionIndex = index;
        break;
      }
    }
    policy.pending = { input, actionIndex };
    $("#selectedAction").innerHTML = "";
    const strong = document.createElement("strong");
    strong.textContent = `Action ${actionIndex + 1}`;
    const explanation = document.createElement("span");
    explanation.textContent = ` selected · probability ${(probabilities[actionIndex] * 100).toFixed(1)}%`;
    $("#selectedAction").append(strong, explanation);
    $("#policyStatus").textContent = "Try this action in your task, then enter the observed reward.";
    $("#submitReward").disabled = false;
    renderActionProbabilities(probabilities);
  } catch (error) {
    announce(error.message);
  }
});
$("#submitReward").addEventListener("click", () => {
  try {
    if (!policy.pending) throw new Error("Choose an action before submitting reward feedback.");
    const reward = Number($("#observedReward").value);
    if (!Number.isFinite(reward)) throw new Error("Enter a finite reward value.");
    const oldBaseline = policy.updates === 0 ? 0 : policy.baseline;
    const update = policyStep(
      network,
      policy.pending.input,
      policy.pending.actionIndex,
      reward,
      oldBaseline,
      network.learningRate,
    );
    policy.baseline = policy.updates === 0 ? reward : 0.9 * policy.baseline + 0.1 * reward;
    policy.updates += 1;
    policy.pending = null;
    $("#policyStatus").textContent = `Policy updated · advantage ${update.advantage.toFixed(3)} · feedback events ${policy.updates}`;
    $("#selectedAction").textContent = "Reward recorded; choose another action to continue.";
    renderPolicy();
    renderEvaluation();
    renderActionProbabilities(network.forward(parseFeatures()));
    saveWorkspace();
  } catch (error) {
    announce(error.message);
  }
});
$("#resetPolicy").addEventListener("click", () => {
  policy = { baseline: 0, updates: 0, pending: null };
  renderPolicy();
  $("#selectedAction").textContent = "No action selected.";
  saveWorkspace();
  announce("Reward-learning state reset.");
});

window.addEventListener("beforeinstallprompt", (event) => {
  event.preventDefault();
  installPrompt = event;
  $("#installButton").hidden = false;
});
$("#installButton").addEventListener("click", async () => {
  if (!installPrompt) return;
  await installPrompt.prompt();
  installPrompt = null;
  $("#installButton").hidden = true;
});

restoreLocalCharacterModel();
loadWorkspace();
render();
setPage(pages.some((page) => page.dataset.pagePanel === location.hash.slice(1)) ? location.hash.slice(1) : "design");
saveWorkspace();
if ("serviceWorker" in navigator && location.protocol !== "file:") {
  navigator.serviceWorker.register("./service-worker.js").catch((error) => {
    $("#saveStatus").textContent = "Offline setup unavailable";
    console.error("Service worker registration failed", error);
  });
}
void refreshLocalModel();
