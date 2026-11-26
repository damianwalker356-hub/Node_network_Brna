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

const STORAGE_KEY = "brna-studio-v1";
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
    let syncError = null;
    try {
      for (const example of examples) {
        await localModelRequest("teach", example);
      }
    } catch (error) {
      syncError = error;
    }
    localModelHealth = await localModelRequest("health");
    const model = localModelHealth.model;
    $("#localModelBadge").textContent = model.trained ? "Model trained" : "Model needs training";
    $("#localModelStatus").textContent =
      `Self-hosted server connected · ${model.training_characters.toLocaleString()} training characters · ` +
      `${localModelHealth.conversation_examples} saved examples. ` +
      "The model and data stay on this computer.";
    $("#trainFromExamples").disabled = localModelHealth.conversation_examples === 0;
    $("#trainFromCorpus").disabled = false;
    if (syncError) {
      $("#modelTrainingStatus").textContent =
        `Server connected, but browser examples could not all sync: ${syncError.message}`;
    }
  } catch (error) {
    localModelHealth = null;
    $("#localModelBadge").textContent = "Server not connected";
    $("#localModelStatus").textContent =
      `Local generation is only available while the downloaded desktop app is running. ` +
      `The GitHub Pages app is a browser interface and cannot host model inference. (${error.message})`;
    $("#trainFromExamples").disabled = true;
    $("#trainFromCorpus").disabled = true;
  }
}

async function trainLocalModel(source, text = undefined) {
  const epochs = Number($("#modelTrainingEpochs").value);
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
    throw new Error("Training epochs must be a whole number from 1 to 50.");
  }
  const button = source === "examples" ? $("#trainFromExamples") : $("#trainFromCorpus");
  button.disabled = true;
  $("#modelTrainingStatus").textContent = "Training the local character model…";
  try {
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
  } finally {
    button.disabled =
      !localModelHealth ||
      (source === "examples" &&
        localModelHealth.conversation_examples === 0 &&
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
  } else {
    response.textContent = "No sufficiently similar taught response was found. Teach a matching example first.";
    response.classList.remove("has-answer");
  }
});
$("#refreshLocalModel").addEventListener("click", refreshLocalModel);
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
    if (!file) throw new Error("Choose a plain-text training corpus first.");
    if (file.size > 50_000) throw new Error("Training text must not exceed 50,000 bytes.");
    const text = await file.text();
    await trainLocalModel("corpus", text);
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
