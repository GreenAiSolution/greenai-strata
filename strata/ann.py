"""Vector indexes: an exact brute-force baseline and a hand-rolled HNSW.

Keeping both is the point. An approximate index is only trustworthy if you can
measure what it lost, so `evaluate.py` reports HNSW recall *against the exact
index* on the same queries. An ANN layer you haven't measured is a silent
recall bug waiting to happen.

The HNSW here is a direct implementation of Malkov & Yashunin (2016):
multi-layer navigable small world graph, level assigned by an exponential
decay, greedy descent through upper layers, `ef`-bounded beam search at layer 0,
and the neighbour-selection heuristic from Algorithm 4 (keep a candidate only if
it is closer to the query than to any already-kept neighbour — this is what
preserves long-range links instead of collapsing into a cluster).
"""

from __future__ import annotations

import heapq
import math
import pickle
from pathlib import Path

import numpy as np


class ExactIndex:
    """Brute-force cosine over normalised vectors. O(n·d), exact by definition."""

    kind = "exact"

    def __init__(self, vectors: np.ndarray):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query: np.ndarray, k: int = 10, **_) -> list[tuple[int, float]]:
        sims = self.vectors @ query.astype(np.float32)
        k = min(k, sims.shape[0])
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]

    def scores(self, query: np.ndarray) -> np.ndarray:
        return self.vectors @ query.astype(np.float32)


class HNSW:
    """Hierarchical Navigable Small World graph over cosine distance."""

    kind = "hnsw"

    def __init__(self, M: int = 16, ef_construction: int = 200, seed: int = 0):
        self.M = M
        self.M0 = M * 2               # layer 0 gets a denser graph
        self.ef_construction = ef_construction
        self.mL = 1.0 / math.log(M)   # level-assignment normalisation factor
        self.rng = np.random.default_rng(seed)
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        self.levels: list[int] = []
        self.graph: list[dict[int, list[int]]] = []
        self.entry: int | None = None
        self.max_level = -1

    # ------------------------------------------------------------- internals
    def _dist(self, query: np.ndarray, node: int) -> float:
        return 1.0 - float(self.vectors[node] @ query)

    def _search_layer(self, query: np.ndarray, entries: list[int], ef: int,
                      level: int) -> list[tuple[float, int]]:
        """Beam search within one layer. Returns [(distance, node)] unsorted."""
        layer = self.graph[level]
        visited = set(entries)
        candidates: list[tuple[float, int]] = []
        found: list[tuple[float, int]] = []   # max-heap keyed on -distance
        for node in entries:
            d = self._dist(query, node)
            heapq.heappush(candidates, (d, node))
            heapq.heappush(found, (-d, node))
        while candidates:
            dist, node = heapq.heappop(candidates)
            if found and dist > -found[0][0]:
                break                          # every remaining option is worse
            for neighbour in layer.get(node, ()):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                d = self._dist(query, neighbour)
                if len(found) < ef or d < -found[0][0]:
                    heapq.heappush(candidates, (d, neighbour))
                    heapq.heappush(found, (-d, neighbour))
                    if len(found) > ef:
                        heapq.heappop(found)
        return [(-neg, node) for neg, node in found]

    def _select_neighbours(self, base: int, candidates: list[tuple[float, int]],
                           m: int) -> list[int]:
        """Algorithm 4: diversity-preserving pruning."""
        ordered = sorted(candidates)
        kept: list[int] = []
        for dist, node in ordered:
            if node == base:
                continue
            if len(kept) >= m:
                break
            # keep `node` only if it is closer to `base` than to anything we
            # already kept — otherwise it is a redundant short link
            if all(dist < 1.0 - float(self.vectors[node] @ self.vectors[other])
                   for other in kept):
                kept.append(node)
        if len(kept) < m:                      # top up with the plain nearest
            for _, node in ordered:
                if node != base and node not in kept:
                    kept.append(node)
                if len(kept) >= m:
                    break
        return kept

    # ------------------------------------------------------------------ build
    def build(self, vectors: np.ndarray, *, progress=None) -> "HNSW":
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n = self.vectors.shape[0]
        self.levels = [
            int(-math.log(max(u, 1e-12)) * self.mL)
            for u in self.rng.random(n)
        ]
        self.max_level = max(self.levels) if n else -1
        self.graph = [dict() for _ in range(self.max_level + 1)]

        for node in range(n):
            self._insert(node)
            if progress and node % 250 == 0:
                progress(node, n)
        return self

    def _insert(self, node: int) -> None:
        level = self.levels[node]
        vector = self.vectors[node]

        if self.entry is None:
            self.entry = node
            for lvl in range(level + 1):
                self.graph[lvl][node] = []
            return

        entries = [self.entry]
        # 1. greedy descent through the layers above the node's own level
        for lvl in range(self.max_level, level, -1):
            if lvl >= len(self.graph):
                continue
            entries = [self._search_layer(vector, entries, 1, lvl)[0][1]]

        # 2. ef-bounded search + linking on every layer the node lives on
        for lvl in range(min(level, self.max_level), -1, -1):
            found = self._search_layer(vector, entries, self.ef_construction, lvl)
            m = self.M0 if lvl == 0 else self.M
            neighbours = self._select_neighbours(node, found, m)
            self.graph[lvl][node] = neighbours
            for neighbour in neighbours:
                links = self.graph[lvl].setdefault(neighbour, [])
                links.append(node)
                if len(links) > m:
                    rescored = [
                        (1.0 - float(self.vectors[neighbour] @ self.vectors[other]), other)
                        for other in links
                    ]
                    self.graph[lvl][neighbour] = self._select_neighbours(
                        neighbour, rescored, m
                    )
            entries = [n for _, n in found] or entries

        if level > self.max_level:
            self.entry = node
            self.max_level = level

    # ----------------------------------------------------------------- search
    def search(self, query: np.ndarray, k: int = 10, ef: int = 64
               ) -> list[tuple[int, float]]:
        if self.entry is None:
            return []
        query = np.ascontiguousarray(query, dtype=np.float32)
        entries = [self.entry]
        for lvl in range(self.max_level, 0, -1):
            entries = [self._search_layer(query, entries, 1, lvl)[0][1]]
        found = self._search_layer(query, entries, max(ef, k), 0)
        found.sort()
        return [(node, 1.0 - dist) for dist, node in found[:k]]

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        with Path(path).open("wb") as fh:
            pickle.dump(
                {
                    "M": self.M, "ef_construction": self.ef_construction,
                    "levels": self.levels, "graph": self.graph,
                    "entry": self.entry, "max_level": self.max_level,
                    "vectors": self.vectors,
                },
                fh, protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "HNSW":
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        obj = cls(M=blob["M"], ef_construction=blob["ef_construction"])
        obj.levels = blob["levels"]
        obj.graph = blob["graph"]
        obj.entry = blob["entry"]
        obj.max_level = blob["max_level"]
        obj.vectors = blob["vectors"]
        obj.M0 = obj.M * 2
        return obj
