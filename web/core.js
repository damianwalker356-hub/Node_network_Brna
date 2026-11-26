export const MAX_LAYER_WIDTH = 32;
export const DEFAULT_DATASET = [
  { input: [0, 0], target: [0] },
  { input: [0, 1], target: [1] },
  { input: [1, 0], target: [1] },
  { input: [1, 1], target: [0] },
];

export function validateArchitecture(sizes) {
  if (!Array.isArray(sizes) || sizes.length < 2 || sizes.length > 8) {
    throw new Error("A network needs an input layer, an output layer, and up to 6 hidden layers.");
  }
  if (sizes.some((size) => !Number.isInteger(size) || size < 1 || size > MAX_LAYER_WIDTH)) {
    throw new Error(`Layer sizes must be whole numbers from 1 to ${MAX_LAYER_WIDTH}.`);
  }
  return sizes.slice();
}

export function countParameters(sizes) {
  validateArchitecture(sizes);
  return sizes.slice(1).reduce((total, width, index) => {
    return total + width * (sizes[index] + 1);
  }, 0);
}

export class DenseNetwork {
  constructor(sizes, { seed = 1, learningRate = 0.12 } = {}) {
    this.kind = "dense";
    this.sizes = validateArchitecture(sizes);
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
      throw new Error("Learning rate must be greater than 0 and no more than 1.");
    }
    this.learningRate = learningRate;
    this._random = makeRandom(seed);
    this.weights = [];
    this.biases = [];
    for (let layer = 1; layer < this.sizes.length; layer += 1) {
      const inputSize = this.sizes[layer - 1];
      const outputSize = this.sizes[layer];
      const scale = Math.sqrt(2 / (inputSize + outputSize));
      this.weights.push(
        Array.from({ length: outputSize }, () =>
          Array.from({ length: inputSize }, () => (this._random() * 2 - 1) * scale),
        ),
      );
      this.biases.push(Array(outputSize).fill(0));
    }
  }

  forward(input) {
    let values = this.#validateInput(input);
    for (let layer = 0; layer < this.weights.length - 1; layer += 1) {
      values = activate(this.weights[layer], this.biases[layer], values, "tanh");
    }
    return activate(
      this.weights.at(-1),
      this.biases.at(-1),
      values,
      this.sizes.at(-1) === 1 ? "sigmoid" : "softmax",
    );
  }

  loss(examples) {
    validateDataset(examples, this.sizes[0], this.sizes.at(-1));
    return examples.reduce((sum, example) => {
      const output = this.forward(example.input);
      return sum + crossEntropy(output, example.target);
    }, 0) / examples.length;
  }

  trainExample(example, learningRate = this.learningRate) {
    validateExample(example, this.sizes[0], this.sizes.at(-1));
    const output = this.forward(example.input);
    const loss = crossEntropy(output, example.target);
    this.applyOutputGradient(
      example.input,
      output.map((value, index) => value - example.target[index]),
      learningRate,
    );
    return loss;
  }

  applyOutputGradient(input, outputGradient, learningRate = this.learningRate) {
    validateGradient(outputGradient, this.sizes.at(-1), learningRate);
    const activations = [this.#validateInput(input)];
    for (let layer = 0; layer < this.weights.length - 1; layer += 1) {
      activations.push(activate(this.weights[layer], this.biases[layer], activations.at(-1), "tanh"));
    }
    let delta = Array.from(outputGradient);
    const deltas = [delta];
    for (let layer = this.weights.length - 2; layer >= 0; layer -= 1) {
      const previous = activations[layer + 1];
      delta = previous.map((value, neuron) => {
        const propagated = this.weights[layer + 1].reduce(
          (sum, nextWeights, nextNeuron) => sum + nextWeights[neuron] * deltas[0][nextNeuron],
          0,
        );
        return propagated * (1 - value * value);
      });
      deltas.unshift(delta);
    }
    applyGradients(this, activations, deltas, learningRate);
  }

  parameterCount() {
    return countParameters(this.sizes);
  }

  trainEpoch(examples, learningRate = this.learningRate) {
    validateDataset(examples, this.sizes[0], this.sizes.at(-1));
    for (const example of examples) this.trainExample(example, learningRate);
    return this.loss(examples);
  }

  genome() {
    return [...this.weights.flat(2), ...this.biases.flat()];
  }

  setGenome(genome) {
    const values = Array.from(genome);
    if (values.length !== countParameters(this.sizes) || values.some((value) => !Number.isFinite(value))) {
      throw new Error("Genome does not match this network's architecture.");
    }
    let offset = 0;
    this.weights = this.weights.map((layer) =>
      layer.map((neuron) => neuron.map(() => values[offset++])),
    );
    this.biases = this.biases.map((layer) => layer.map(() => values[offset++]));
  }

  toJSON() {
    return {
      version: 1,
      kind: "dense",
      sizes: this.sizes,
      learningRate: this.learningRate,
      genome: this.genome(),
    };
  }

  static fromJSON(data) {
    if (!data || data.version !== 1 || (data.kind && data.kind !== "dense") || !Array.isArray(data.sizes)) {
      throw new Error("Unsupported or invalid network file.");
    }
    const network = new DenseNetwork(data.sizes, { learningRate: data.learningRate });
    network.setGenome(data.genome);
    return network;
  }

  #validateInput(input) {
    const values = Array.from(input ?? []);
    if (
      values.length !== this.sizes[0] ||
      values.some((value) => !Number.isFinite(value))
    ) {
      throw new Error(`Input must contain ${this.sizes[0]} finite values.`);
    }
    return values;
  }
}

export class MessagePassingNetwork {
  constructor(
    {
      nodes = 12,
      features = 4,
      hidden = 16,
      hops = 3,
      readoutNode = nodes - 1,
      indegree = 4,
      outputs = 1,
      communication = "learned",
      adjacency = null,
    } = {},
    { seed = 1, learningRate = 0.01 } = {},
  ) {
    this.kind = "message-passing";
    this.config = validateMessageConfig({
      nodes,
      features,
      hidden,
      hops,
      readoutNode,
      indegree,
      outputs,
      communication,
    });
    const { nodes: nodeCount, features: featureCount, hidden: width, indegree: degree } = this.config;
    this.adjacency = adjacency
      ? validateAdjacency(adjacency, nodeCount, degree)
      : makeAdjacency(nodeCount, degree, seed);
    this.sizes = [nodeCount * featureCount, width, outputs];
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
      throw new Error("Learning rate must be greater than 0 and no more than 1.");
    }
    this.learningRate = learningRate;
    const random = makeRandom(seed ^ 0x9e3779b9);
    const scale = Math.sqrt(2 / (featureCount + width));
    this.Win = randomMatrix(featureCount, width, random, scale);
    this.E = randomMatrix(nodeCount, width, random, 0.1);
    this.Ws = randomMatrix(width, width, random, Math.sqrt(1 / width));
    this.Wm = randomMatrix(width, width, random, Math.sqrt(1 / width));
    this.bias = Array(width).fill(0);
    this.outputWeights = randomMatrix(outputs, width, random, Math.sqrt(1 / width));
    this.outputBias = Array(outputs).fill(0);
    this.gates = Array.from({ length: nodeCount }, () => Array(nodeCount).fill(0));
  }

  forward(input) {
    return this.#forwardCached(input).output;
  }

  loss(examples) {
    validateDataset(examples, this.sizes[0], this.sizes.at(-1));
    return examples.reduce((sum, example) => {
      return sum + crossEntropy(this.forward(example.input), example.target);
    }, 0) / examples.length;
  }

  trainExample(example, learningRate = this.learningRate) {
    validateExample(example, this.sizes[0], this.sizes.at(-1));
    const output = this.forward(example.input);
    const loss = crossEntropy(output, example.target);
    this.applyOutputGradient(
      example.input,
      output.map((value, index) => value - example.target[index]),
      learningRate,
    );
    return loss;
  }

  applyOutputGradient(input, outputGradient, learningRate = this.learningRate) {
    validateGradient(outputGradient, this.sizes.at(-1), learningRate);
    const cache = this.#forwardCached(input);
    const nodeCount = this.config.nodes;
    const width = this.config.hidden;
    const outputCount = this.config.outputs;
    const outputWeightsGradient = Array.from({ length: outputCount }, (_, action) =>
      cache.states.at(-1)[this.config.readoutNode].map((value) => value * outputGradient[action]),
    );
    const outputBiasGradient = Array.from(outputGradient);
    let stateGradient = Array.from({ length: nodeCount }, () => Array(width).fill(0));
    for (let neuron = 0; neuron < width; neuron += 1) {
      for (let action = 0; action < outputCount; action += 1) {
        stateGradient[this.config.readoutNode][neuron] +=
          this.outputWeights[action][neuron] * outputGradient[action];
      }
    }

    const wsGradient = matrix(width, width);
    const wmGradient = matrix(width, width);
    const biasGradient = Array(width).fill(0);
    const gateGradient = matrix(nodeCount, nodeCount);
    for (let hop = this.config.hops; hop > 0; hop -= 1) {
      const state = cache.states[hop];
      const previous = cache.states[hop - 1];
      const messages = cache.messages[hop - 1];
      const aggregate = cache.aggregates[hop - 1];
      const preActivationGradient = matrix(nodeCount, width);
      for (let node = 0; node < nodeCount; node += 1) {
        for (let neuron = 0; neuron < width; neuron += 1) {
          const gradient = stateGradient[node][neuron] * (1 - state[node][neuron] ** 2);
          preActivationGradient[node][neuron] = gradient;
          biasGradient[neuron] += gradient;
          for (let inputNeuron = 0; inputNeuron < width; inputNeuron += 1) {
            wsGradient[inputNeuron][neuron] += previous[node][inputNeuron] * gradient;
            wmGradient[inputNeuron][neuron] += aggregate[node][inputNeuron] * gradient;
          }
        }
      }

      const aggregateGradient = matrix(nodeCount, width);
      for (let node = 0; node < nodeCount; node += 1) {
        for (let inputNeuron = 0; inputNeuron < width; inputNeuron += 1) {
          for (let neuron = 0; neuron < width; neuron += 1) {
            aggregateGradient[node][inputNeuron] +=
              preActivationGradient[node][neuron] * this.Wm[inputNeuron][neuron];
          }
        }
      }

      const previousGradient = matrix(nodeCount, width);
      for (let receiver = 0; receiver < nodeCount; receiver += 1) {
        for (let sender = 0; sender < nodeCount; sender += 1) {
          const activeGate = cache.activeGates[receiver][sender];
          for (let neuron = 0; neuron < width; neuron += 1) {
            if (this.config.communication !== "off") {
              gateGradient[receiver][sender] +=
                aggregateGradient[receiver][neuron] *
                messages[sender][neuron] *
                activeGate *
                (1 - activeGate);
            }
            previousGradient[sender][neuron] += activeGate * aggregateGradient[receiver][neuron];
          }
        }
        for (let neuron = 0; neuron < width; neuron += 1) {
          for (let inputNeuron = 0; inputNeuron < width; inputNeuron += 1) {
            previousGradient[receiver][inputNeuron] +=
              preActivationGradient[receiver][neuron] * this.Ws[inputNeuron][neuron];
          }
        }
      }
      stateGradient = previousGradient;
    }

    const inputGradient = matrix(this.config.features, width);
    const embeddingGradient = matrix(nodeCount, width);
    const inputMatrix = reshapeInput(input, nodeCount, this.config.features);
    for (let node = 0; node < nodeCount; node += 1) {
      for (let neuron = 0; neuron < width; neuron += 1) {
        const gradient = stateGradient[node][neuron] * (1 - cache.states[0][node][neuron] ** 2);
        embeddingGradient[node][neuron] = gradient;
        for (let feature = 0; feature < this.config.features; feature += 1) {
          inputGradient[feature][neuron] += inputMatrix[node][feature] * gradient;
        }
      }
    }

    subtractScaled(this.Win, inputGradient, learningRate);
    subtractScaled(this.E, embeddingGradient, learningRate);
    subtractScaled(this.Ws, wsGradient, learningRate);
    subtractScaled(this.Wm, wmGradient, learningRate);
    subtractScaled(this.outputWeights, outputWeightsGradient, learningRate);
    subtractScaled(this.gates, gateGradient, learningRate, this.adjacency);
    subtractVector(this.bias, biasGradient, learningRate);
    subtractVector(this.outputBias, outputBiasGradient, learningRate);
  }

  trainEpoch(examples, learningRate = this.learningRate) {
    validateDataset(examples, this.sizes[0], this.sizes.at(-1));
    for (const example of examples) this.trainExample(example, learningRate);
    return this.loss(examples);
  }

  parameterCount() {
    return (
      this.config.features * this.config.hidden +
      this.config.nodes * this.config.hidden +
      2 * this.config.hidden ** 2 +
      this.config.hidden +
      this.config.outputs * (this.config.hidden + 1) +
      this.adjacency.flat().reduce((sum, connected) => sum + connected, 0)
    );
  }

  genome() {
    const activeGateWeights = [];
    for (let receiver = 0; receiver < this.config.nodes; receiver += 1) {
      for (let sender = 0; sender < this.config.nodes; sender += 1) {
        if (this.adjacency[receiver][sender]) activeGateWeights.push(this.gates[receiver][sender]);
      }
    }
    return [
      ...this.Win.flat(),
      ...this.E.flat(),
      ...this.Ws.flat(),
      ...this.Wm.flat(),
      ...this.bias,
      ...this.outputWeights.flat(),
      ...this.outputBias,
      ...activeGateWeights,
    ];
  }

  setGenome(genome) {
    const values = Array.from(genome ?? []);
    if (values.length !== this.parameterCount() || values.some((value) => !Number.isFinite(value))) {
      throw new Error("Genome does not match this network's architecture.");
    }
    let offset = 0;
    this.Win = readMatrix(values, offset, this.config.features, this.config.hidden);
    offset += this.config.features * this.config.hidden;
    this.E = readMatrix(values, offset, this.config.nodes, this.config.hidden);
    offset += this.config.nodes * this.config.hidden;
    this.Ws = readMatrix(values, offset, this.config.hidden, this.config.hidden);
    offset += this.config.hidden ** 2;
    this.Wm = readMatrix(values, offset, this.config.hidden, this.config.hidden);
    offset += this.config.hidden ** 2;
    this.bias = values.slice(offset, offset + this.config.hidden);
    offset += this.config.hidden;
    this.outputWeights = readMatrix(values, offset, this.config.outputs, this.config.hidden);
    offset += this.config.outputs * this.config.hidden;
    this.outputBias = values.slice(offset, offset + this.config.outputs);
    offset += this.config.outputs;
    this.gates = matrix(this.config.nodes, this.config.nodes);
    for (let receiver = 0; receiver < this.config.nodes; receiver += 1) {
      for (let sender = 0; sender < this.config.nodes; sender += 1) {
        if (this.adjacency[receiver][sender]) this.gates[receiver][sender] = values[offset++];
      }
    }
  }

  toJSON() {
    return {
      version: 1,
      kind: this.kind,
      config: this.config,
      adjacency: this.adjacency,
      learningRate: this.learningRate,
      genome: this.genome(),
    };
  }

  static fromJSON(data) {
    if (
      !data ||
      data.version !== 1 ||
      data.kind !== "message-passing" ||
      !data.config ||
      !Array.isArray(data.adjacency)
    ) {
      throw new Error("Unsupported or invalid message-passing model file.");
    }
    const network = new MessagePassingNetwork(
      { ...data.config, adjacency: data.adjacency },
      {
        learningRate: data.learningRate,
      },
    );
    network.setGenome(data.genome);
    return network;
  }

  #forwardCached(input) {
    const inputMatrix = reshapeInput(input, this.config.nodes, this.config.features);
    const activeGates = this.config.communication === "off"
      ? matrix(this.config.nodes, this.config.nodes)
      : this.adjacency.map((row, receiver) =>
          row.map((connected, sender) => connected * sigmoid(this.gates[receiver][sender])),
        );
    const states = [matrix(this.config.nodes, this.config.hidden)];
    for (let node = 0; node < this.config.nodes; node += 1) {
      for (let neuron = 0; neuron < this.config.hidden; neuron += 1) {
        let value = this.E[node][neuron];
        for (let feature = 0; feature < this.config.features; feature += 1) {
          value += inputMatrix[node][feature] * this.Win[feature][neuron];
        }
        states[0][node][neuron] = Math.tanh(value);
      }
    }
    const messages = [];
    const aggregates = [];
    for (let hop = 0; hop < this.config.hops; hop += 1) {
      const message = states.at(-1).map((row) =>
        this.config.communication === "quantized"
          ? row.map((value) => Math.round(value * 3) / 3)
          : row.slice(),
      );
      const aggregate = matrix(this.config.nodes, this.config.hidden);
      for (let receiver = 0; receiver < this.config.nodes; receiver += 1) {
        for (let sender = 0; sender < this.config.nodes; sender += 1) {
          for (let neuron = 0; neuron < this.config.hidden; neuron += 1) {
            aggregate[receiver][neuron] += activeGates[receiver][sender] * message[sender][neuron];
          }
        }
      }
      const next = matrix(this.config.nodes, this.config.hidden);
      for (let node = 0; node < this.config.nodes; node += 1) {
        for (let neuron = 0; neuron < this.config.hidden; neuron += 1) {
          let value = this.bias[neuron];
          for (let other = 0; other < this.config.hidden; other += 1) {
            value +=
              states.at(-1)[node][other] * this.Ws[other][neuron] +
              aggregate[node][other] * this.Wm[other][neuron];
          }
          next[node][neuron] = Math.tanh(value);
        }
      }
      messages.push(message);
      aggregates.push(aggregate);
      states.push(next);
    }
    const logits = this.outputWeights.map((weights, action) =>
      weights.reduce(
        (sum, weight, neuron) => sum + weight * states.at(-1)[this.config.readoutNode][neuron],
        this.outputBias[action],
      ),
    );
    const output = this.config.outputs === 1 ? [sigmoid(logits[0])] : softmax(logits);
    return { states, messages, aggregates, activeGates, output };
  }
}

export function createNetworkFromJSON(data) {
  if (data?.kind === "message-passing") return MessagePassingNetwork.fromJSON(data);
  return DenseNetwork.fromJSON(data);
}

export function evolveGeneration(network, examples, population, options = {}) {
  const { eliteFraction = 0.2, mutationScale = 0.15, seed = 1 } = options;
  if (!Array.isArray(population) || population.length < 2) {
    throw new Error("Evolution needs at least two candidate networks.");
  }
  if (
    !Number.isFinite(eliteFraction) ||
    eliteFraction <= 0 ||
    eliteFraction > 1 ||
    !Number.isFinite(mutationScale) ||
    mutationScale < 0
  ) {
    throw new Error("Evolution settings are outside their valid range.");
  }
  const random = makeRandom(seed);
  const ranked = population
    .map((genome, index) => {
      const candidate = createNetworkFromJSON(network.toJSON());
      candidate.setGenome(genome);
      return { genome: Array.from(genome), fitness: -candidate.loss(examples), index };
    })
    .sort((left, right) => right.fitness - left.fitness || left.index - right.index);
  const eliteCount = Math.max(1, Math.ceil(population.length * eliteFraction));
  const elites = ranked.slice(0, eliteCount).map(({ genome }) => genome);
  const next = elites.map((genome) => genome.slice());
  while (next.length < population.length) {
    const parentA = elites[Math.floor(random() * eliteCount)];
    const parentB = elites[Math.floor(random() * eliteCount)];
    next.push(
      parentA.map((value, index) => {
        const inherited = random() < 0.5 ? value : parentB[index];
        return inherited + gaussian(random) * mutationScale;
      }),
    );
  }
  return {
    population: next,
    bestGenome: ranked[0].genome.slice(),
    bestFitness: ranked[0].fitness,
  };
}

export function initialPopulation(network, size, { seed = 1, mutationScale = 0.25 } = {}) {
  if (!Number.isInteger(size) || size < 2 || size > 200) {
    throw new Error("Population size must be from 2 to 200.");
  }
  if (!Number.isFinite(mutationScale) || mutationScale < 0 || mutationScale > 2) {
    throw new Error("Mutation scale must be finite and between 0 and 2.");
  }
  const random = makeRandom(seed);
  const base = network.genome();
  return [
    base,
    ...Array.from({ length: size - 1 }, () =>
      base.map((value) => value + gaussian(random) * mutationScale),
    ),
  ];
}

export function policyStep(network, input, actionIndex, reward, baseline, learningRate) {
  if (network.sizes.at(-1) < 2) {
    throw new Error("Policy learning requires at least two output actions.");
  }
  if (!Number.isInteger(actionIndex) || actionIndex < 0 || actionIndex >= network.sizes.at(-1)) {
    throw new Error("Selected action is not an output of this network.");
  }
  if (![reward, baseline, learningRate].every(Number.isFinite) || learningRate <= 0) {
    throw new Error("Reward, baseline, and learning rate must be finite; learning rate must be positive.");
  }
  const probabilities = network.forward(input);
  const advantage = Math.max(-5, Math.min(5, reward - baseline));
  const outputGradient = probabilities.map(
    (probability, index) => advantage * (probability - Number(index === actionIndex)),
  );
  network.applyOutputGradient(input, outputGradient, learningRate);
  return { probabilities, advantage };
}

export function retrieveReply(examples, prompt, threshold = 0.25) {
  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error("Similarity threshold must be between 0 and 1.");
  }
  const query = tokenize(prompt);
  if (query.size === 0) return null;
  let best = null;
  for (const example of examples) {
    const candidate = tokenize(example.prompt);
    const union = new Set([...query, ...candidate]);
    const overlap = [...query].filter((token) => candidate.has(token)).length;
    const score = union.size === 0 ? 0 : overlap / union.size;
    if (!best || score > best.score) best = { response: example.response, score };
  }
  return best && best.score >= threshold ? best : null;
}

export function evaluateBinaryAccuracy(network, examples) {
  if (network.sizes.at(-1) !== 1) return null;
  return examples.reduce(
    (correct, example) =>
      correct + Number(Number(network.forward(example.input)[0] >= 0.5) === Number(example.target[0] >= 0.5)),
    0,
  ) / examples.length;
}

function applyGradients(network, activations, deltas, learningRate) {
  for (let layer = 0; layer < network.weights.length; layer += 1) {
    for (let neuron = 0; neuron < network.weights[layer].length; neuron += 1) {
      network.biases[layer][neuron] -= learningRate * deltas[layer][neuron];
      for (let inputIndex = 0; inputIndex < network.weights[layer][neuron].length; inputIndex += 1) {
        network.weights[layer][neuron][inputIndex] -=
          learningRate * deltas[layer][neuron] * activations[layer][inputIndex];
      }
    }
  }
}

function activate(weights, biases, input, mode) {
  const values = weights.map(
    (neuron, index) => neuron.reduce((sum, value, inputIndex) => sum + value * input[inputIndex], biases[index]),
  );
  if (mode === "tanh") return values.map(Math.tanh);
  if (mode === "sigmoid") return values.map(sigmoid);
  return softmax(values);
}

function crossEntropy(output, target) {
  const outputs = Array.isArray(output) ? output : [output];
  return -target.reduce(
    (sum, value, index) =>
      sum + value * Math.log(Math.max(outputs[index], 1e-12)) +
      (outputs.length === 1 ? (1 - value) * Math.log(Math.max(1 - outputs[index], 1e-12)) : 0),
    0,
  );
}

function validateGradient(gradient, outputSize, learningRate) {
  if (
    !Array.isArray(gradient) ||
    gradient.length !== outputSize ||
    gradient.some((value) => !Number.isFinite(value))
  ) {
    throw new Error(`Output gradient must contain ${outputSize} finite values.`);
  }
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) {
    throw new Error("Learning rate must be greater than 0 and no more than 1.");
  }
}

function validateMessageConfig(config) {
  const integerFields = [
    ["nodes", 2, 32],
    ["features", 1, 8],
    ["hidden", 1, 32],
    ["hops", 0, 6],
    ["outputs", 1, 16],
  ];
  for (const [field, minimum, maximum] of integerFields) {
    if (!Number.isInteger(config[field]) || config[field] < minimum || config[field] > maximum) {
      throw new Error(`${field} must be a whole number from ${minimum} to ${maximum}.`);
    }
  }
  if (
    !Number.isInteger(config.readoutNode) ||
    config.readoutNode < 0 ||
    config.readoutNode >= config.nodes
  ) {
    throw new Error("Readout node must identify a node in the network.");
  }
  if (
    !Number.isInteger(config.indegree) ||
    config.indegree < 1 ||
    config.indegree >= config.nodes
  ) {
    throw new Error("Each node needs between 1 and node-count minus 1 incoming edges.");
  }
  if (!["learned", "quantized", "off"].includes(config.communication)) {
    throw new Error("Communication must be learned, quantized, or off.");
  }
  return { ...config };
}

function validateAdjacency(adjacency, nodes, indegree) {
  if (
    !Array.isArray(adjacency) ||
    adjacency.length !== nodes ||
    adjacency.some(
      (row, receiver) =>
        !Array.isArray(row) ||
        row.length !== nodes ||
        row.some(
          (edge, sender) =>
            !Number.isInteger(edge) ||
            (edge !== 0 && edge !== 1) ||
            (receiver === sender && edge !== 0),
        ) ||
        row.reduce((sum, edge) => sum + edge, 0) !== indegree,
    )
  ) {
    throw new Error("Adjacency must match the configured in-degree and have no self edges.");
  }
  return adjacency.map((row) => row.slice());
}

function makeAdjacency(nodes, indegree, seed) {
  const random = makeRandom((seed ^ 0xa5a5a5a5) >>> 0);
  return Array.from({ length: nodes }, (_, receiver) => {
    const sources = Array.from({ length: nodes }, (_, sender) => sender).filter(
      (sender) => sender !== receiver,
    );
    for (let index = sources.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(random() * (index + 1));
      [sources[index], sources[swap]] = [sources[swap], sources[index]];
    }
    const row = Array(nodes).fill(0);
    for (const sender of sources.slice(0, indegree)) row[sender] = 1;
    return row;
  });
}

function randomMatrix(rows, columns, random, scale) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: columns }, () => (random() * 2 - 1) * scale),
  );
}

function matrix(rows, columns) {
  return Array.from({ length: rows }, () => Array(columns).fill(0));
}

function readMatrix(values, offset, rows, columns) {
  return Array.from({ length: rows }, (_, row) =>
    values.slice(offset + row * columns, offset + (row + 1) * columns),
  );
}

function reshapeInput(input, nodes, features) {
  if (
    !Array.isArray(input) ||
    input.length !== nodes * features ||
    input.some((value) => !Number.isFinite(value))
  ) {
    throw new Error(`Input must contain ${nodes * features} finite node-feature values.`);
  }
  return Array.from({ length: nodes }, (_, node) =>
    input.slice(node * features, (node + 1) * features),
  );
}

function subtractScaled(target, gradient, learningRate, mask = null) {
  for (let row = 0; row < target.length; row += 1) {
    for (let column = 0; column < target[row].length; column += 1) {
      if (!mask || mask[row][column]) {
        target[row][column] -= learningRate * gradient[row][column];
      }
    }
  }
}

function subtractVector(target, gradient, learningRate) {
  for (let index = 0; index < target.length; index += 1) {
    target[index] -= learningRate * gradient[index];
  }
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
}

function softmax(values) {
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

function validateDataset(examples, inputSize, outputSize) {
  if (!Array.isArray(examples) || examples.length === 0) {
    throw new Error("Training dataset must contain at least one example.");
  }
  for (const example of examples) validateExample(example, inputSize, outputSize);
}

function validateExample(example, inputSize, outputSize) {
  if (
    !example ||
    !Array.isArray(example.input) ||
    !Array.isArray(example.target) ||
    example.input.length !== inputSize ||
    example.target.length !== outputSize ||
    [...example.input, ...example.target].some((value) => !Number.isFinite(value))
  ) {
    throw new Error(`Each example needs ${inputSize} finite inputs and ${outputSize} finite targets.`);
  }
  if (outputSize === 1) {
    if (example.target[0] < 0 || example.target[0] > 1) {
      throw new Error("Binary targets must be between 0 and 1.");
    }
  } else if (
    example.target.some((value) => value < 0) ||
    Math.abs(example.target.reduce((sum, value) => sum + value, 0) - 1) > 1e-6
  ) {
    throw new Error("Multi-class targets must be probabilities that sum to 1.");
  }
}

function tokenize(text) {
  return new Set(String(text).toLocaleLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);
}

function gaussian(random) {
  const first = Math.max(random(), Number.EPSILON);
  const second = random();
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}

function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
