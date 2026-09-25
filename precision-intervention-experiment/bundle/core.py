from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@dataclass(frozen=True)
class GraphData:
    x: torch.Tensor
    y: torch.Tensor
    edge_src: torch.Tensor
    edge_dst: torch.Tensor
    degree: torch.Tensor
    graph_index: torch.Tensor
    graph_id: str
    condition: str
    environment: tuple[float, float, float]

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def undirected_edges(self) -> int:
        return int(self.edge_src.numel() // 2)


def _geometric_skip_positions(total: int, probability: float, rng: np.random.Generator) -> np.ndarray:
    if total < 0:
        raise ValueError("total must be nonnegative")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must lie in [0,1]")
    if total == 0 or probability == 0.0:
        return np.empty(0, dtype=np.int64)
    if probability == 1.0:
        return np.arange(total, dtype=np.int64)
    position_chunks: list[np.ndarray] = []
    current = -1
    while True:
        remaining = total - current - 1
        expected_successes = remaining * probability
        chunk_size = int(min(65536, max(64, math.ceil(1.25 * expected_successes + 16))))
        jumps = rng.geometric(probability, size=chunk_size).astype(np.int64, copy=False)
        candidates = current + np.cumsum(jumps, dtype=np.int64)
        valid = candidates[candidates < total]
        if valid.size:
            position_chunks.append(valid)
        if valid.size < candidates.size:
            break
        current = int(candidates[-1])
    return np.concatenate(position_chunks) if position_chunks else np.empty(0, dtype=np.int64)


def sample_undirected_gnp(size: int, probability: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sample the strict lower triangle of G(size, probability) exactly."""
    if size < 0:
        raise ValueError("size must be nonnegative")
    total = size * (size - 1) // 2
    positions = _geometric_skip_positions(total, probability, rng)
    if positions.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    # Row v contains the v lower-triangular entries (v, 0),...,(v, v-1).
    v = np.floor((1.0 + np.sqrt(8.0 * positions + 1.0)) / 2.0).astype(np.int64)
    starts = v * (v - 1) // 2
    w = positions - starts
    if not np.all((0 <= w) & (w < v) & (v < size)):
        raise RuntimeError("lower-triangle index inversion failed")
    return v, w.astype(np.int64, copy=False)


def sample_bipartite_gnp(
    left_size: int,
    right_size: int,
    probability: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if left_size < 0 or right_size < 0:
        raise ValueError("block sizes must be nonnegative")
    positions = _geometric_skip_positions(left_size * right_size, probability, rng)
    if positions.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return positions // right_size, positions % right_size


def edge_probabilities(n: int, lambda_: float, rho: float) -> tuple[float, float]:
    if n < 4 or n % 2:
        raise ValueError("n must be even and at least four")
    if lambda_ <= 0:
        raise ValueError("lambda must be positive")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0,1)")
    same = lambda_ * (1.0 + rho) / (n - 2)
    cross = lambda_ * (1.0 - rho) / n
    if same > 1.0 or cross > 1.0:
        raise ValueError("declared environment produces an invalid edge probability")
    return same, cross


def generate_graph(
    *,
    n: int,
    feature_dimension: int,
    lambda_: float,
    rho: float,
    mu: float,
    seed: int,
    graph_id: str,
    condition: str,
) -> GraphData:
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be positive")
    rng = np.random.default_rng(seed)
    labels = np.concatenate([
        np.ones(n // 2, dtype=np.int8),
        -np.ones(n // 2, dtype=np.int8),
    ])
    labels = labels[rng.permutation(n)]
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == -1)
    p_same, p_cross = edge_probabilities(n, lambda_, rho)

    pp_a, pp_b = sample_undirected_gnp(positive.size, p_same, rng)
    nn_a, nn_b = sample_undirected_gnp(negative.size, p_same, rng)
    pn_a, pn_b = sample_bipartite_gnp(positive.size, negative.size, p_cross, rng)
    left = np.concatenate([positive[pp_a], negative[nn_a], positive[pn_a]])
    right = np.concatenate([positive[pp_b], negative[nn_b], negative[pn_b]])
    undirected_src = np.concatenate([left, right]).astype(np.int64, copy=False)
    undirected_dst = np.concatenate([right, left]).astype(np.int64, copy=False)
    order = np.lexsort((undirected_src, undirected_dst))
    edge_src = undirected_src[order]
    edge_dst = undirected_dst[order]
    degree = np.bincount(edge_dst, minlength=n).astype(np.int64, copy=False)

    features = rng.normal(size=(n, feature_dimension)).astype(np.float32)
    features[:, 0] += (mu * labels).astype(np.float32)
    targets = (labels == 1).astype(np.float32)
    return GraphData(
        x=torch.from_numpy(features),
        y=torch.from_numpy(targets),
        edge_src=torch.from_numpy(edge_src),
        edge_dst=torch.from_numpy(edge_dst),
        degree=torch.from_numpy(degree),
        graph_index=torch.zeros(n, dtype=torch.int64),
        graph_id=graph_id,
        condition=condition,
        environment=(float(lambda_), float(rho), float(mu)),
    )


def combine_graphs(graphs: Sequence[GraphData], graph_id: str) -> GraphData:
    if not graphs:
        raise ValueError("at least one graph is required")
    feature_dimension = graphs[0].x.shape[1]
    if any(graph.x.shape[1] != feature_dimension for graph in graphs):
        raise ValueError("feature dimensions do not match")
    x_parts, y_parts, src_parts, dst_parts, degree_parts, graph_parts = [], [], [], [], [], []
    offset = 0
    for graph_number, graph in enumerate(graphs):
        x_parts.append(graph.x)
        y_parts.append(graph.y)
        src_parts.append(graph.edge_src + offset)
        dst_parts.append(graph.edge_dst + offset)
        degree_parts.append(graph.degree)
        graph_parts.append(torch.full((graph.n,), graph_number, dtype=torch.int64))
        offset += graph.n
    return GraphData(
        x=torch.cat(x_parts),
        y=torch.cat(y_parts),
        edge_src=torch.cat(src_parts),
        edge_dst=torch.cat(dst_parts),
        degree=torch.cat(degree_parts),
        graph_index=torch.cat(graph_parts),
        graph_id=graph_id,
        condition=graphs[0].condition,
        environment=graphs[0].environment,
    )


def to_device(graph: GraphData, device: torch.device) -> GraphData:
    return GraphData(
        x=graph.x.to(device),
        y=graph.y.to(device),
        edge_src=graph.edge_src.to(device),
        edge_dst=graph.edge_dst.to(device),
        degree=graph.degree.to(device),
        graph_index=graph.graph_index.to(device),
        graph_id=graph.graph_id,
        condition=graph.condition,
        environment=graph.environment,
    )


def graph_seed(experiment_id: str, block_seed: int, pool: str, condition: str, graph_index: int) -> int:
    return stable_seed(experiment_id, "graph", block_seed, pool, condition, graph_index)


def graph_identifier(experiment_id: str, block_seed: int, pool: str, condition: str, graph_index: int) -> str:
    seed = graph_seed(experiment_id, block_seed, pool, condition, graph_index)
    return f"{pool}:{block_seed}:{condition}:{graph_index}:{seed}"


def generate_pool(
    config: dict,
    *,
    block_seed: int,
    pool: str,
    condition: dict,
    count: int,
    n_override: int | None = None,
) -> list[GraphData]:
    experiment_id = config["experiment_id"]
    n = int(n_override or config["generator"]["n"])
    dimension = int(config["generator"]["feature_dimension"])
    result = []
    for graph_index in range(count):
        seed = graph_seed(experiment_id, block_seed, pool, condition["condition"], graph_index)
        identifier = graph_identifier(experiment_id, block_seed, pool, condition["condition"], graph_index)
        result.append(generate_graph(
            n=n,
            feature_dimension=dimension,
            lambda_=float(condition["lambda"]),
            rho=float(condition["rho"]),
            mu=float(condition["mu"]),
            seed=seed,
            graph_id=identifier,
            condition=condition["condition"],
        ))
    return result


def source_degree_normalization(graphs: Iterable[GraphData]) -> tuple[float, float, float]:
    values = torch.cat([torch.log1p(graph.degree.to(torch.float64)) for graph in graphs])
    location = float(values.mean())
    scale = float(values.std(unbiased=True))
    delta = location
    if not math.isfinite(scale) or scale < 1e-8 or delta <= 0.0:
        raise RuntimeError("invalid source degree normalization")
    return location, scale, delta


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def degree_bin_label(lower: int, upper: int | None) -> str:
    return f"{lower}-{upper}" if upper is not None else f"{lower}+"
