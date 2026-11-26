const MAX_TRAINING_CHARACTERS = 50_000;
const MAX_MODEL_CHARACTERS = 50_000;
const CONTEXT_SIZE = 4;

export class LocalCharacterModel {
  constructor() {
    this.trainingText = "";
    this.trainingCharacters = 0;
    this.transitions = new Map();
  }

  get trained() {
    return this.trainingCharacters > 0;
  }

  train(text, epochs = 3) {
    if (typeof text !== "string") throw new TypeError("training text must be a string");
    const characters = Array.from(text);
    if (characters.length > MAX_TRAINING_CHARACTERS) {
      throw new RangeError(`training text must not exceed ${MAX_TRAINING_CHARACTERS} characters`);
    }
    if (characters.length < 2) throw new RangeError("training text must contain at least two characters");
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 50) {
      throw new RangeError("epochs must be a whole number from 1 to 50");
    }

    const separator = this.trainingText ? "\n" : "";
    this.trainingText = Array.from(this.trainingText + separator + text)
      .slice(-MAX_MODEL_CHARACTERS)
      .join("");
    this.trainingCharacters += characters.length * epochs;
    this.#buildTransitions();
    return { trainingCharacters: this.trainingCharacters, epochs };
  }

  generate(prompt, maxNewTokens = 240) {
    if (typeof prompt !== "string" || !prompt.trim()) throw new TypeError("prompt must not be empty");
    if (!this.trained) throw new Error("model has not been trained yet");
    if (!Number.isInteger(maxNewTokens) || maxNewTokens < 1 || maxNewTokens > 1024) {
      throw new RangeError("max_new_tokens must be a whole number from 1 to 1024");
    }

    const context = Array.from(prompt).slice(-CONTEXT_SIZE);
    const generated = [];
    for (let index = 0; index < maxNewTokens; index += 1) {
      let candidates;
      for (let length = Math.min(CONTEXT_SIZE, context.length); length >= 0; length -= 1) {
        const key = length === 0 ? "" : context.slice(-length).join("");
        candidates = this.transitions.get(key);
        if (candidates) break;
      }
      if (!candidates) break;
      const total = candidates.total;
      let draw = Math.random() * total;
      let next = "";
      for (const [character, count] of candidates.counts) {
        draw -= count;
        if (draw < 0) {
          next = character;
          break;
        }
      }
      if (!next) break;
      generated.push(next);
      context.push(next);
      if (context.length > CONTEXT_SIZE) context.shift();
    }
    return generated.join("").trim();
  }

  toJSON() {
    return {
      version: 1,
      trainingText: this.trainingText,
      trainingCharacters: this.trainingCharacters,
    };
  }

  static fromJSON(value) {
    if (
      !value ||
      value.version !== 1 ||
      typeof value.trainingText !== "string" ||
      Array.from(value.trainingText).length > MAX_MODEL_CHARACTERS ||
      !Number.isSafeInteger(value.trainingCharacters) ||
      value.trainingCharacters < 0
    ) {
      throw new TypeError("saved browser model is invalid");
    }
    const model = new LocalCharacterModel();
    model.trainingText = value.trainingText;
    model.trainingCharacters = value.trainingCharacters;
    if (model.trained && Array.from(model.trainingText).length < 2) {
      throw new TypeError("saved browser model has no training corpus");
    }
    model.#buildTransitions();
    return model;
  }

  #buildTransitions() {
    const characters = Array.from(this.trainingText);
    const transitions = new Map();
    for (let index = 0; index < characters.length; index += 1) {
      for (let size = 0; size <= Math.min(CONTEXT_SIZE, index); size += 1) {
        const context = characters.slice(index - size, index).join("");
        let candidates = transitions.get(context);
        if (!candidates) {
          candidates = { counts: new Map(), total: 0 };
          transitions.set(context, candidates);
        }
        const character = characters[index];
        candidates.counts.set(character, (candidates.counts.get(character) ?? 0) + 1);
        candidates.total += 1;
      }
    }
    this.transitions = transitions;
  }
}
