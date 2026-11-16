"""Нейросеть с нуля на numpy: полносвязная сеть, backprop и Adam написаны руками."""
import numpy as np


class MLP:
    def __init__(self, sizes, seed=0, dropout=0.3):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.dropout = dropout
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0                                    # счётчик шагов Adam

    # ---------- прямой проход ----------
    def forward(self, X, train=False, rng=None):
        acts, zs, masks = [X], [], []
        h = X
        last = len(self.W) - 1
        for i in range(len(self.W)):
            z = h @ self.W[i] + self.b[i]
            if i == last:
                return z, acts, zs, masks
            zs.append(z)
            h = np.maximum(z, 0)
            m = None
            if train and self.dropout > 0:
                m = (rng.random(h.shape) >= self.dropout).astype(np.float32) / (1 - self.dropout)
                h = h * m
            masks.append(m)
            acts.append(h)

    def predict_proba(self, X):
        logits = self.forward(np.asarray(X, np.float32))[0]
        s = logits - logits.max(1, keepdims=True)
        p = np.exp(s)
        return p / p.sum(1, keepdims=True)

    # ---------- обратный проход ----------
    def loss_and_grads(self, X, y, rng, l2=1e-4):
        logits, acts, zs, masks = self.forward(X, True, rng)
        n = len(y)
        s = logits - logits.max(1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(1, keepdims=True)
        loss = float(-np.log(p[np.arange(n), y] + 1e-9).mean())
        dz = p
        dz[np.arange(n), y] -= 1
        dz /= n
        L = len(self.W)
        gW, gb = [None] * L, [None] * L
        for i in reversed(range(L)):
            gW[i] = acts[i].T @ dz + l2 * self.W[i]
            gb[i] = dz.sum(0)
            if i > 0:
                dh = dz @ self.W[i].T
                if masks[i - 1] is not None:
                    dh = dh * masks[i - 1]
                dz = dh * (zs[i - 1] > 0)
        return loss, gW, gb

    def step(self, gW, gb, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        for P, G, M, V in ((self.W, gW, self.mW, self.vW), (self.b, gb, self.mb, self.vb)):
            for i in range(len(P)):
                M[i] = b1 * M[i] + (1 - b1) * G[i]
                V[i] = b2 * V[i] + (1 - b2) * G[i] ** 2
                P[i] -= lr * (M[i] / c1) / (np.sqrt(V[i] / c2) + eps)

    # ---------- сохранение ----------
    def weights_dict(self):
        d = {}
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            d[f"W{i}"], d[f"b{i}"] = w, b
        return d

    def load_weights(self, d):
        for i in range(len(self.W)):
            self.W[i] = np.array(d[f"W{i}"], dtype=np.float32)
            self.b[i] = np.array(d[f"b{i}"], dtype=np.float32)

    def state_arrays(self):
        d = self.weights_dict()
        for i in range(len(self.W)):
            d[f"mW{i}"], d[f"vW{i}"] = self.mW[i], self.vW[i]
            d[f"mb{i}"], d[f"vb{i}"] = self.mb[i], self.vb[i]
        d["adam_t"] = np.array(self.t, dtype=np.int64)
        return d

    def load_state(self, d):
        self.load_weights(d)
        for i in range(len(self.W)):
            self.mW[i], self.vW[i] = np.array(d[f"mW{i}"]), np.array(d[f"vW{i}"])
            self.mb[i], self.vb[i] = np.array(d[f"mb{i}"]), np.array(d[f"vb{i}"])
        self.t = int(d["adam_t"])
