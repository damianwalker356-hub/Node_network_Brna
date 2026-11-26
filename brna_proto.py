
"""B-RNA v0 trainable prototype (Rev C). Pure numpy, manual backprop. No network I/O (BRNA-R-7.2).
Trains a 12-node message-passing network with learned edge gates on a distributed-interaction task,
and compares it with ablations and single-model baselines (BRNA-R-7.3: all outcomes are reported)."""
import numpy as np, json, hashlib, time, sys

N, DX = 12, 4
def make_data(n, rng):
    X = rng.normal(size=(n, N, DX))
    u = X[:, :, 0]
    s = sum(u[:, 2*k] * u[:, 2*k+1] for k in range(6))
    return X, (s > 0).astype(float)

def make_adj(rng, indeg=4):
    A = np.zeros((N, N))
    for i in range(N):
        A[i, rng.choice([j for j in range(N) if j != i], indeg, replace=False)] = 1
    return A

class Adam:
    def __init__(s, ps, lr=0.01): s.ps, s.lr, s.m, s.v, s.t = ps, lr, [np.zeros_like(p) for p in ps], [np.zeros_like(p) for p in ps], 0
    def step(s, gs):
        s.t += 1
        for p, g, m, v in zip(s.ps, gs, s.m, s.v):
            m *= .9; m += .1 * g; v *= .999; v += .001 * g * g
            p -= s.lr * (m / (1 - .9**s.t)) / (np.sqrt(v / (1 - .999**s.t)) + 1e-8)

sig = lambda z: 1 / (1 + np.exp(-z))

class NodeNet:
    def __init__(s, rng, d=16, T=3, mode="vector", out=11):
        s.d, s.T, s.mode, s.out = d, T, mode, out
        s.adj = make_adj(rng)
        g = lambda *sh, sc=1.0: rng.normal(0, sc / np.sqrt(sh[0]), sh)
        s.Win, s.E, s.Ws, s.Wm = g(DX, d), rng.normal(0, .1, (N, d)), g(d, d), g(d, d)
        s.b, s.wo, s.bo, s.G = np.zeros(d), g(d), np.zeros(1), np.zeros((N, N))
        s.ps = [s.Win, s.E, s.Ws, s.Wm, s.b, s.wo, s.bo, s.G]
        s.opt = Adam(s.ps)
    def nparams(s): return sum(p.size for p in s.ps[:-1]) + int(s.adj.sum())
    def q(s, H): return np.round(H * 3) / 3 if s.mode == "quantized" else H
    def gate(s): return 0 * s.adj if s.mode == "nocomm" else s.adj * sig(s.G)
    def fwd(s, X):
        Ag = s.gate(); H = [np.tanh(X @ s.Win + s.E)]; M, AG = [], []
        for _ in range(s.T):
            m = s.q(H[-1]); agg = np.einsum('ij,bjd->bid', Ag, m)
            M.append(m); AG.append(agg); H.append(np.tanh(H[-1] @ s.Ws + agg @ s.Wm + s.b))
        return H, M, AG, Ag, H[-1][:, s.out] @ s.wo + s.bo
    def predict(s, X): return s.fwd(X)[4] > 0
    def train_step(s, X, y):
        H, M, AG, Ag, lo = s.fwd(X); B = len(X); p = sig(lo); dl = (p - y) / B
        gWs, gWm, gb = np.zeros_like(s.Ws), np.zeros_like(s.Wm), np.zeros_like(s.b)
        gwo = H[-1][:, s.out].T @ dl; gbo = np.array([dl.sum()])
        dH = np.zeros_like(H[-1]); dH[:, s.out] = dl[:, None] * s.wo; dAg = np.zeros((N, N))
        for t in range(s.T, 0, -1):
            dZ = dH * (1 - H[t]**2); gb += dZ.sum((0, 1))
            gWs += np.einsum('bnd,bne->de', H[t-1], dZ); gWm += np.einsum('bnd,bne->de', AG[t-1], dZ)
            dagg = dZ @ s.Wm.T
            dAg += np.einsum('bid,bjd->ij', dagg, M[t-1])
            dH = dZ @ s.Ws.T + np.einsum('ij,bid->bjd', Ag, dagg)
        dZ0 = dH * (1 - H[0]**2); gWin = np.einsum('bnx,bnd->xd', X, dZ0); gE = dZ0.sum(0)
        sg = sig(s.G); gG = dAg * s.adj * sg * (1 - sg)
        if s.mode == "nocomm": gG = 0 * gG
        s.opt.step([gWin, gE, gWs, gWm, gb, gwo, gbo, gG])

class MLP:
    def __init__(s, rng, h):
        s.W1, s.b1, s.w2, s.b2 = rng.normal(0, 1/np.sqrt(N*DX), (N*DX, h)), np.zeros(h), rng.normal(0, 1/np.sqrt(h), h), np.zeros(1)
        s.ps = [s.W1, s.b1, s.w2, s.b2]; s.opt = Adam(s.ps)
    def nparams(s): return sum(p.size for p in s.ps)
    def predict(s, X): return (np.tanh(X.reshape(len(X), -1) @ s.W1 + s.b1) @ s.w2 + s.b2) > 0
    def train_step(s, X, y):
        x = X.reshape(len(X), -1); h = np.tanh(x @ s.W1 + s.b1); p = sig(h @ s.w2 + s.b2); dl = (p - y) / len(X)
        dh = dl[:, None] * s.w2 * (1 - h**2)
        s.opt.step([x.T @ dh, dh.sum(0), h.T @ dl, np.array([dl.sum()])])

def run(kind, seed, steps=1500, batch=256):
    rng = np.random.default_rng(seed)
    m = {"vector": lambda: NodeNet(rng, mode="vector"), "quantized": lambda: NodeNet(rng, mode="quantized"),
         "nocomm": lambda: NodeNet(rng, mode="nocomm"), "single_small": lambda: MLP(rng, 12), "single_large": lambda: MLP(rng, 64)}[kind]()
    for _ in range(steps): m.train_step(*make_data(batch, rng))
    Xt, yt = make_data(5000, np.random.default_rng(10_000 + seed))
    return dict(kind=kind, seed=seed, params=int(m.nparams()), test_acc=float((m.predict(Xt) == yt).mean()))

def run_record(results, extra=None):
    src = open(__file__, "rb").read() if "__file__" in globals() else b""
    return dict(code_sha256=hashlib.sha256(src).hexdigest(), time=time.strftime("%Y-%m-%dT%H:%M:%S"), results=results, **(extra or {}))
