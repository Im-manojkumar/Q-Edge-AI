"""Classical baselines that every quantum path is compared against."""
from __future__ import annotations

import math
import time

import numpy as np


def maxcut_greedy(n: int, edges) -> tuple[str, float]:
    side = [0] * n
    adj = [[] for _ in range(n)]
    for i, j, w in edges:
        adj[i].append((j, w))
        adj[j].append((i, w))
    for v in range(n):
        gain0 = sum(w for u, w in adj[v] if u < v and side[u] == 1)
        gain1 = sum(w for u, w in adj[v] if u < v and side[u] == 0)
        side[v] = 1 if gain1 > gain0 else 0
    bits = "".join(map(str, side))
    return bits, sum(w for i, j, w in edges if bits[i] != bits[j])


def maxcut_annealing(n: int, edges, sweeps: int = 400, seed: int = 0) -> tuple[str, float]:
    rng = np.random.default_rng(seed)
    adj = [[] for _ in range(n)]
    for i, j, w in edges:
        adj[i].append((j, w))
        adj[j].append((i, w))
    s = rng.integers(0, 2, n)
    cur = sum(w for i, j, w in edges if s[i] != s[j])
    best, best_s = cur, s.copy()
    for t in range(sweeps):
        T = 2.0 * (1 - t / sweeps) + 1e-3
        for v in rng.permutation(n):
            delta = sum(w if s[u] == s[v] else -w for u, w in adj[v])
            if delta > 0 or rng.random() < math.exp(delta / T):
                s[v] ^= 1
                cur += delta
                if cur > best:
                    best, best_s = cur, s.copy()
    return "".join(map(str, best_s)), float(best)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0
