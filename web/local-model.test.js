import assert from "node:assert/strict";
import test from "node:test";

import { LocalCharacterModel } from "./local-model.js";

test("browser model trains from text and generates only learned characters", () => {
  const model = new LocalCharacterModel();

  const trainingText = "abcabcabc 🧠";
  const result = model.train(trainingText, 2);
  const generated = model.generate("abc", 32);

  assert.equal(result.trainingCharacters, Array.from(trainingText).length * 2);
  assert.ok(generated.length > 0);
  assert.ok(Array.from(generated).every((character) => model.trainingText.includes(character)));
});

test("browser model preserves its local training data when restored", () => {
  const model = new LocalCharacterModel();
  model.train("abcabcabc", 3);

  const restored = LocalCharacterModel.fromJSON(JSON.parse(JSON.stringify(model)));

  assert.equal(restored.trained, true);
  assert.equal(restored.trainingText, model.trainingText);
  assert.equal(restored.trainingCharacters, model.trainingCharacters);
  assert.ok(restored.generate("abc", 12).length > 0);
});

test("browser model rejects invalid training and generation requests", () => {
  const model = new LocalCharacterModel();

  assert.throws(() => model.generate("hello"), /not been trained/);
  assert.throws(() => model.train("x"), /at least two/);
  assert.throws(() => model.train("valid text", 0), /epochs/);
  model.train("valid training text");
  assert.throws(() => model.generate("hello", 0), /max_new_tokens/);
  assert.throws(() => LocalCharacterModel.fromJSON({ version: 1 }), /invalid/);
});
