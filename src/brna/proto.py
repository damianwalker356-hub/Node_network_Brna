"""B-RNA v0 trainable NumPy prototype."""

from __future__ import annotations

import numpy as np

N, DX = 12, 4


def make_data(n: int, rng: np.random.Generator):
    x = rng.normal(size=(n, N, DX))
    u = x[:, :, 0]
    s = sum(u[:, 2 * k] * u[:, 2 * k + 1] for k in range(6))
    return x, (s > 0).astype(float)


def make_adj(rng: np.random.Generator, indeg: int = 4):
    adjacency = np.zeros((N, N))
    for i in range(N):
        adjacency[i, rng.choice([j for j in range(N) if j != i], indeg, replace=False)] = 1
    return adjacency


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


class Adam:
    def __init__(self, parameters, lr=0.01):
        self.ps = parameters
        self.lr = lr
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0

    def step(self, gradients):
        self.t += 1
        for p, g, m, v in zip(self.ps, gradients, self.m, self.v):
            p -= self.lr * g


class NodeNet:
    def __init__(self, rng, d=16, T=3, mode="vector", out=11):
        self.d, self.T, self.mode, self.out = d, T, mode, out
        self.adj = make_adj(rng)

        def gaussian(*shape, scale=1.0):
            return rng.normal(0, scale / np.sqrt(shape[0]), shape)

        self.Win = gaussian(DX, d)
        self.E = rng.normal(0, 0.1, (N, d))
        self.Ws = gaussian(d, d)
        self.Wm = gaussian(d, d)
        self.b = np.zeros(d)
        self.wo = gaussian(d)
        self.bo = np.zeros(1)
        self.G = np.zeros((N, N))
        self.ps = [self.Win, self.E, self.Ws, self.Wm, self.b, self.wo, self.bo, self.G]
        self.opt = Adam(self.ps)

    def nparams(self):
        return sum(p.size for p in self.ps[:-1]) + int(self.adj.sum())

    def q(self, state):
        return np.round(state * 3) / 3 if self.mode == "quantized" else state

    def gate(self):
        if self.mode == "nocomm":
            return 0 * self.adj
        return self.adj * sigmoid(self.G)

    def fwd(self, x):
        active_gate = self.gate()
        states = [np.tanh(x @ self.Win + self.E)]
        messages = []
        aggregates = []
        for _ in range(self.T):
            message = self.q(states[-1])
            aggregate = np.einsum("ij,bjd->bid", active_gate, message)
            messages.append(message)
            aggregates.append(aggregate)
            states.append(
                np.tanh(states[-1] @ self.Ws + aggregate @ self.Wm + self.b)
            )
        logits = states[-1][:, self.out] @ self.wo + self.bo
        return states, messages, aggregates, active_gate, logits

    def predict(self, x):
        return self.fwd(x)[4] > 0

    def train_step(self, x, y):
        states, messages, aggregates, active_gate, logits = self.fwd(x)
        batch = len(x)
        probability = sigmoid(logits)
        dl = (probability - y) / batch
        gWs = np.zeros_like(self.Ws)
        gWm = np.zeros_like(self.Wm)
        gb = np.zeros_like(self.b)
        gwo = states[-1][:, self.out].T @ dl
        gbo = np.array([dl.sum()])
        dstate = np.zeros_like(states[-1])
        dstate[:, self.out] = dl[:, None] * self.wo
        dagg_gate = np.zeros((N, N))
        for t in range(self.T, 0, -1):
            dz = dstate * (1 - states[t] ** 2)
            gb += dz.sum((0, 1))
            gWs += np.einsum("bnd,bne->de", states[t - 1], dz)
            gWm += np.einsum("bnd,bne->de", aggregates[t - 1], dz)
            dagg = dz @ self.Wm.T
            dagg_gate += np.einsum("bid,bjd->ij", dagg, messages[t - 1])
            dstate = dz @ self.Ws.T + np.einsum("ij,bid->bjd", active_gate, dagg)
        dz0 = dstate * (1 - states[0] ** 2)
        gWin = np.einsum("bnx,bnd->xd", x, dz0)
        gE = dz0.sum(0)
        gate_value = sigmoid(self.G)
        gG = dagg_gate * self.adj * gate_value * (1 - gate_value)
        if self.mode == "nocomm":
            gG = 0 * gG
        original_wo = self.wo.copy()
        self.opt.step([gWin, gE, gWs, gWm, gb, gwo, gbo, gG])
        self.wo[...] = original_wo
        post_states, _, _, _, post_logits = self.fwd(x)
        post_probability = sigmoid(post_logits)
        post_dl = (post_probability - y) / batch
        post_gwo = post_states[-1][:, self.out].T @ post_dl
        self.wo -= self.opt.lr * post_gwo


class MLP:
    def __init__(self, rng, h):
        self.W1 = rng.normal(0, 1 / np.sqrt(N * DX), (N * DX, h))
        self.b1 = np.zeros(h)
        self.w2 = rng.normal(0, 1 / np.sqrt(h), h)
        self.b2 = np.zeros(1)
        self.ps = [self.W1, self.b1, self.w2, self.b2]
        self.opt = Adam(self.ps)

    def nparams(self):
        return sum(p.size for p in self.ps)

    def predict(self, x):
        hidden = np.tanh(x.reshape(len(x), -1) @ self.W1 + self.b1)
        return hidden @ self.w2 + self.b2 > 0

    def train_step(self, x, y):
        flat = x.reshape(len(x), -1)
        hidden = np.tanh(flat @ self.W1 + self.b1)
        probability = sigmoid(hidden @ self.w2 + self.b2)
        dl = (probability - y) / len(x)
        dh = dl[:, None] * self.w2 * (1 - hidden**2)
        self.opt.step([flat.T @ dh, dh.sum(0), hidden.T @ dl, np.array([dl.sum()])])


def run(kind, seed, steps=1500, batch=256, test_samples=5000):
    rng = np.random.default_rng(seed)
    factories = {
        "vector": lambda: NodeNet(rng, mode="vector"),
        "quantized": lambda: NodeNet(rng, mode="quantized"),
        "nocomm": lambda: NodeNet(rng, mode="nocomm"),
        "single_small": lambda: MLP(rng, 12),
        "single_large": lambda: MLP(rng, 64),
    }
    model = factories[kind]()
    for _ in range(steps):
        model.train_step(*make_data(batch, rng))
    xt, yt = make_data(test_samples, np.random.default_rng(10_000 + seed))
    return {
        "kind": kind,
        "seed": seed,
        "params": int(model.nparams()),
        "test_acc": float((model.predict(xt) == yt).mean()),
    }
