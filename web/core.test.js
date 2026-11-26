import assert from "node:assert/strict";
import test from "node:test";

import {
  DEFAULT_DATASET,
  DenseNetwork,
  MessagePassingNetwork,
  createNetworkFromJSON,
  countParameters,
  evaluateBinaryAccuracy,
  evolveGeneration,
  initialPopulation,
  policyStep,
  retrieveReply,
} from "./core.js";

test("architecture parameter count and model JSON round-trip", () => {
  const network = new DenseNetwork([2, 3, 1], { seed: 17 });
  assert.equal(countParameters(network.sizes), 13);
  const restored = DenseNetwork.fromJSON(network.toJSON());
  assert.deepEqual(restored.sizes, [2, 3, 1]);
  assert.deepEqual(restored.genome(), network.genome());
  assert.deepEqual(restored.forward([0, 1]), network.forward([0, 1]));
});

test("supervised gradient training lowers XOR loss", () => {
  const network = new DenseNetwork([2, 6, 1], { seed: 9, learningRate: 0.15 });
  const start = network.loss(DEFAULT_DATASET);
  for (let epoch = 0; epoch < 1500; epoch += 1) network.trainEpoch(DEFAULT_DATASET);
  assert.ok(network.loss(DEFAULT_DATASET) < start * 0.2);
  assert.ok(evaluateBinaryAccuracy(network, DEFAULT_DATASET) >= 0.75);
});

test("BRNA message-passing default matches the prototype and round-trips", () => {
  const network = new MessagePassingNetwork({}, { seed: 11 });
  assert.equal(network.parameterCount(), 849);
  assert.equal(network.forward(Array(48).fill(0)).length, 1);
  const restored = createNetworkFromJSON(network.toJSON());
  assert.deepEqual(restored.config, network.config);
  assert.deepEqual(restored.adjacency, network.adjacency);
  assert.deepEqual(restored.genome(), network.genome());
  assert.deepEqual(restored.forward(Array(48).fill(0)), network.forward(Array(48).fill(0)));
});

test("message-passing networks train, evolve, and learn from reward feedback", () => {
  const network = new MessagePassingNetwork(
    { nodes: 2, features: 1, hidden: 5, hops: 1, readoutNode: 1, indegree: 1, outputs: 1 },
    { seed: 8, learningRate: 0.05 },
  );
  const examples = [
    { input: [1, 0], target: [1] },
    { input: [0, 1], target: [0] },
  ];
  const startingLoss = network.loss(examples);
  for (let epoch = 0; epoch < 100; epoch += 1) network.trainEpoch(examples);
  assert.ok(network.loss(examples) < startingLoss / 2);

  const population = initialPopulation(network, 4, { seed: 5 });
  const result = evolveGeneration(network, examples, population, { seed: 6 });
  assert.equal(result.population.length, 4);
  assert.equal(result.bestGenome.length, network.parameterCount());

  const policyNetwork = new MessagePassingNetwork(
    { nodes: 2, features: 1, hidden: 4, hops: 1, readoutNode: 1, indegree: 1, outputs: 2 },
    { seed: 4, learningRate: 0.05 },
  );
  const input = [0.2, 0.8];
  const before = policyNetwork.forward(input)[0];
  for (let update = 0; update < 30; update += 1) {
    policyStep(policyNetwork, input, 0, 1, 0, 0.05);
  }
  assert.ok(policyNetwork.forward(input)[0] > before);
});

test("communication-off ablation leaves learned gate weights unchanged", () => {
  const network = new MessagePassingNetwork(
    { nodes: 3, features: 1, hidden: 3, hops: 1, readoutNode: 2, indegree: 1, communication: "off" },
    { seed: 3 },
  );
  const genome = network.genome();
  genome.fill(0.2, genome.length - 3);
  network.setGenome(genome);
  const gateWeights = network.gates.map((row) => row.slice());
  network.trainExample({ input: [1, 0, 0], target: [1] });
  assert.deepEqual(network.gates, gateWeights);
});

test("invalid inputs and datasets fail explicitly", () => {
  const network = new DenseNetwork([2, 1]);
  assert.throws(() => network.forward([1]), /Input must contain 2/);
  assert.throws(() => network.trainEpoch([]), /at least one example/);
  assert.throws(() => new DenseNetwork([2, 33]), /Layer sizes/);
});

test("evolution preserves best candidate and reproduces deterministically", () => {
  const network = new DenseNetwork([2, 3, 1], { seed: 5 });
  const firstPopulation = initialPopulation(network, 8, { seed: 22 });
  const secondPopulation = initialPopulation(network, 8, { seed: 22 });
  assert.deepEqual(firstPopulation, secondPopulation);
  const options = { seed: 77, eliteFraction: 0.25, mutationScale: 0.05 };
  const first = evolveGeneration(network, DEFAULT_DATASET, firstPopulation, options);
  const second = evolveGeneration(network, DEFAULT_DATASET, secondPopulation, options);
  assert.deepEqual(first.population, second.population);
  assert.deepEqual(first.population[0], first.bestGenome);
  assert.ok(first.bestFitness <= 0);
});

test("policy learning moves probability toward a rewarded action", () => {
  const network = new DenseNetwork([2, 5, 2], { seed: 4, learningRate: 0.05 });
  const input = [0.2, 0.8];
  const before = network.forward(input)[0];
  for (let update = 0; update < 60; update += 1) {
    policyStep(network, input, 0, 1, 0, 0.05);
  }
  assert.ok(network.forward(input)[0] > before);
});

test("conversation learning retrieves taught responses and does not invent answers", () => {
  const examples = [{ prompt: "How do networks learn?", response: "From examples and feedback." }];
  assert.equal(retrieveReply(examples, "How do networks learn?").response, examples[0].response);
  assert.equal(retrieveReply(examples, "Where is my blue bicycle?"), null);
});
