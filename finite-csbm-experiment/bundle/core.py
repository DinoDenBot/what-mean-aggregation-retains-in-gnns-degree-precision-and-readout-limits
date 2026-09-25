from __future__ import annotations

import copy
import hashlib
import math
import random
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn

from models import LocalAggregator, NuisanceRegressor


DIRECTIONS = ("rate", "homophily", "context")


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**31 - 1)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def condition_quantities(n_vertices: int, lambda0: float, rho0: float) -> dict[str, float]:
    if n_vertices % 2:
        raise ValueError("balanced finite CSBM requires even n")
    same_candidates = n_vertices // 2 - 1
    cross_candidates = n_vertices // 2
    q_same = lambda0 * (1.0 + rho0) / (2.0 * same_candidates)
    q_cross = lambda0 * (1.0 - rho0) / (2.0 * cross_candidates)
    if not (0.0 < q_same < 1.0 and 0.0 < q_cross < 1.0):
        raise ValueError("invalid Bernoulli probability")
    return {
        "same_candidates": float(same_candidates),
        "cross_candidates": float(cross_candidates),
        "q_same": q_same,
        "q_cross": q_cross,
    }


def raw_fisher_matrix(n_vertices: int, lambda0: float, rho0: float) -> torch.Tensor:
    values = condition_quantities(n_vertices, lambda0, rho0)
    fisher = torch.zeros((3, 3), dtype=torch.float64)
    for sign, mkey, qkey in (
        (1.0, "same_candidates", "q_same"),
        (-1.0, "cross_candidates", "q_cross"),
    ):
        candidates = values[mkey]
        q = values[qkey]
        dq_lambda = q / lambda0
        dq_rho = sign * q / (1.0 + sign * rho0)
        weight = candidates / (q * (1.0 - q))
        derivative = torch.tensor([dq_lambda, dq_rho], dtype=torch.float64)
        fisher[:2, :2] += weight * torch.outer(derivative, derivative)
    fisher[2, 2] = lambda0
    return fisher


def symmetric_power(matrix: torch.Tensor, exponent: float) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(0.5 * (matrix + matrix.T))
    if float(values.min()) <= 0:
        raise ValueError("matrix must be positive definite")
    return (vectors * values.pow(exponent).unsqueeze(0)) @ vectors.T


def canonical_direction_vectors(fisher: torch.Tensor) -> torch.Tensor:
    """Columns are raw-Fisher-unit coordinate tangents expressed against Z."""
    square_root = symmetric_power(fisher, 0.5)
    columns = []
    for index in range(fisher.shape[0]):
        tangent = torch.zeros(fisher.shape[0], dtype=torch.float64)
        tangent[index] = 1.0
        columns.append(square_root @ tangent / torch.sqrt(fisher[index, index]))
    return torch.stack(columns, dim=1)


@dataclass
class RootPool:
    messages: torch.Tensor
    mask: torch.Tensor
    score: torch.Tensor
    raw_score: torch.Tensor
    sample_ids: torch.Tensor

    def __len__(self) -> int:
        return int(self.score.shape[0])


def _binomial(size: int, trials: int, probability: float, generator: torch.Generator) -> torch.Tensor:
    totals = torch.full((size,), float(trials), dtype=torch.float64)
    probabilities = torch.full((size,), float(probability), dtype=torch.float64)
    return torch.binomial(totals, probabilities, generator=generator).to(torch.long)


def generate_root_pool(
    size: int,
    n_vertices: int,
    lambda0: float,
    rho0: float,
    mu0: float,
    seed: int,
    id_offset: int,
) -> RootPool:
    values = condition_quantities(n_vertices, lambda0, rho0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_same = _binomial(size, int(values["same_candidates"]), values["q_same"], generator)
    n_cross = _binomial(size, int(values["cross_candidates"]), values["q_cross"], generator)
    counts = n_same + n_cross
    max_count = max(1, int(counts.max()))
    positions = torch.arange(max_count).unsqueeze(0)
    mask = positions < counts.unsqueeze(1)
    labels = torch.where(
        positions < n_same.unsqueeze(1),
        torch.ones((size, max_count), dtype=torch.float32),
        -torch.ones((size, max_count), dtype=torch.float32),
    )
    labels = labels * mask.to(torch.float32)
    noise = torch.randn((size, max_count), generator=generator, dtype=torch.float32)
    contexts = (mu0 * labels + noise) * mask.to(torch.float32)
    messages = torch.stack(
        [mask.to(torch.float32), labels, contexts, labels * contexts],
        dim=-1,
    )

    centered_same = n_same.to(torch.float64) - values["same_candidates"] * values["q_same"]
    centered_cross = n_cross.to(torch.float64) - values["cross_candidates"] * values["q_cross"]
    rate_score = (
        centered_same / (lambda0 * (1.0 - values["q_same"]))
        + centered_cross / (lambda0 * (1.0 - values["q_cross"]))
    )
    homophily_score = (
        centered_same / ((1.0 + rho0) * (1.0 - values["q_same"]))
        - centered_cross / ((1.0 - rho0) * (1.0 - values["q_cross"]))
    )
    context_score = (labels * contexts).to(torch.float64).sum(1) - mu0 * counts.to(torch.float64)
    raw_score = torch.stack([rate_score, homophily_score, context_score], dim=1)
    inverse_root = symmetric_power(raw_fisher_matrix(n_vertices, lambda0, rho0), -0.5)
    score = raw_score @ inverse_root
    sample_ids = torch.arange(id_offset, id_offset + size, dtype=torch.int64)
    return RootPool(messages, mask, score, raw_score, sample_ids)


def batches(size: int, batch_size: int, seed: int) -> Iterable[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    while True:
        yield torch.randint(0, size, (batch_size,), generator=generator)


@torch.no_grad()
def evaluate_architecture(model: nn.Module, pool: RootPool, device: torch.device, batch_size: int = 1024) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(pool), batch_size):
        sl = slice(start, min(len(pool), start + batch_size))
        prediction = model(pool.messages[sl].to(device), pool.mask[sl].to(device))[0]
        target = pool.score[sl].to(device=device, dtype=prediction.dtype)
        total += torch.square(prediction - target).sum().item()
        count += target.numel()
    return total / max(count, 1)


def fit_architecture(
    name: str,
    lambda0: float,
    train: RootPool,
    validation: RootPool,
    config: dict,
    seed: int,
    device: torch.device,
    maximum_steps: int | None = None,
) -> tuple[LocalAggregator, dict]:
    set_determinism(seed)
    model = LocalAggregator(name, lambda0).to(device)
    settings = config["model"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    stream = batches(len(train), int(settings["batch_size"]), stable_seed(seed, "architecture-batches"))
    steps = int(maximum_steps or settings["maximum_steps"])
    best = math.inf
    best_state = None
    stale = 0
    curve = []
    for step in range(1, steps + 1):
        index = next(stream)
        messages = train.messages[index].to(device)
        mask = train.mask[index].to(device)
        target = train.score[index].to(device=device, dtype=messages.dtype)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(messages, mask)[0]
        loss = torch.square(prediction - target).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip_norm"]))
        optimizer.step()
        if step % int(settings["validation_interval_steps"]) == 0 or step == steps:
            validation_mse = evaluate_architecture(model, validation, device)
            curve.append({"step": step, "train_mse": float(loss), "validation_mse": validation_mse})
            if validation_mse < best - float(settings["early_stopping_minimum_improvement"]):
                best = validation_mse
                stale = 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
                if stale >= int(settings["early_stopping_patience_checks"]):
                    break
    if best_state is None:
        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    return model.to(device).eval(), {"best_validation_mse": best, "curve": curve}


@torch.no_grad()
def encode_pool(model: LocalAggregator, pool: RootPool, device: torch.device, batch_size: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
    aggregates = []
    hidden = []
    model.eval()
    for start in range(0, len(pool), batch_size):
        sl = slice(start, min(len(pool), start + batch_size))
        _, aggregate, representation = model(pool.messages[sl].to(device), pool.mask[sl].to(device))
        aggregates.append(aggregate.cpu())
        hidden.append(representation.cpu())
    return torch.cat(aggregates), torch.cat(hidden)


@torch.no_grad()
def nuisance_mse(model: nn.Module, x: torch.Tensor, z: torch.Tensor, device: torch.device, batch_size: int = 2048) -> float:
    total = 0.0
    count = 0
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        sl = slice(start, min(x.shape[0], start + batch_size))
        prediction = model(x[sl].to(device)).cpu().to(torch.float64)
        total += torch.square(prediction - z[sl]).sum().item()
        count += z[sl].numel()
    return total / max(count, 1)


def fit_nuisance(
    x_fit: torch.Tensor,
    z_fit: torch.Tensor,
    x_validation: torch.Tensor,
    z_validation: torch.Tensor,
    config: dict,
    seed: int,
    device: torch.device,
    maximum_steps: int | None = None,
) -> tuple[NuisanceRegressor, dict]:
    set_determinism(seed)
    settings = config["nuisance_regression"]
    location = x_fit.to(torch.float64).mean(0)
    raw_scale = x_fit.to(torch.float64).std(0, unbiased=True)
    minimum = float(settings["input_standardization"]["minimum_scale"])
    scale = torch.where(raw_scale < minimum, torch.ones_like(raw_scale), raw_scale)
    model = NuisanceRegressor(x_fit.shape[1], location, scale).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    stream = batches(x_fit.shape[0], int(settings["batch_size"]), stable_seed(seed, "nuisance-batches"))
    steps = int(maximum_steps or settings["maximum_steps"])
    best = math.inf
    best_state = None
    stale = 0
    curve = []
    for step in range(1, steps + 1):
        index = next(stream)
        xx = x_fit[index].to(device)
        zz = z_fit[index].to(device=device, dtype=xx.dtype)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xx)
        loss = torch.square(prediction - zz).mean()
        loss.backward()
        optimizer.step()
        if step % int(settings["validation_interval_steps"]) == 0 or step == steps:
            validation_mse = nuisance_mse(model, x_validation, z_validation, device)
            curve.append({"step": step, "train_mse": float(loss), "validation_mse": validation_mse})
            if validation_mse < best - float(settings["early_stopping_minimum_improvement"]):
                best = validation_mse
                stale = 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale += 1
                if stale >= int(settings["early_stopping_patience_checks"]):
                    break
    if best_state is None:
        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.load_state_dict(best_state)
    return model.to(device).eval(), {"best_validation_mse": best, "curve": curve}


@torch.no_grad()
def predict_nuisance(model: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int = 2048) -> torch.Tensor:
    output = []
    model.eval()
    for start in range(0, x.shape[0], batch_size):
        output.append(model(x[start:start + batch_size].to(device)).cpu().to(torch.float64))
    return torch.cat(output)


def orthogonal_fisher(mhat: torch.Tensor, score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    m = mhat.to(torch.float64)
    z = score.to(torch.float64)
    per_matrix = m.unsqueeze(2) * z.unsqueeze(1) + z.unsqueeze(2) * m.unsqueeze(1) - m.unsqueeze(2) * m.unsqueeze(1)
    estimate = 0.5 * (per_matrix.mean(0) + per_matrix.mean(0).T)
    residual = float(torch.square(z - m).sum(1).mean())
    return estimate, per_matrix, residual


def canonical_contributions(per_matrix: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ki,nkl,li->ni", vectors, per_matrix, vectors)


def analytic_score_from_aggregate(
    name: str,
    aggregate: torch.Tensor,
    n_vertices: int,
    lambda0: float,
    rho0: float,
    mu0: float,
) -> torch.Tensor | None:
    if name == "sum":
        n = aggregate[:, 0].to(torch.float64)
        label_sum = aggregate[:, 1].to(torch.float64)
        label_context_sum = aggregate[:, 3].to(torch.float64)
    elif name == "mean_degree":
        n = aggregate[:, 4].to(torch.float64) * math.sqrt(lambda0) + lambda0
        n = torch.where(aggregate[:, 5].bool(), torch.zeros_like(n), n)
        label_sum = aggregate[:, 1].to(torch.float64) * n
        label_context_sum = aggregate[:, 3].to(torch.float64) * n
    else:
        return None
    n_same = 0.5 * (n + label_sum)
    n_cross = 0.5 * (n - label_sum)
    values = condition_quantities(n_vertices, lambda0, rho0)
    centered_same = n_same - values["same_candidates"] * values["q_same"]
    centered_cross = n_cross - values["cross_candidates"] * values["q_cross"]
    rate = (
        centered_same / (lambda0 * (1.0 - values["q_same"]))
        + centered_cross / (lambda0 * (1.0 - values["q_cross"]))
    )
    homophily = (
        centered_same / ((1.0 + rho0) * (1.0 - values["q_same"]))
        - centered_cross / ((1.0 - rho0) * (1.0 - values["q_cross"]))
    )
    context = label_context_sum - mu0 * n
    raw = torch.stack([rate, homophily, context], dim=1)
    inverse_root = symmetric_power(raw_fisher_matrix(n_vertices, lambda0, rho0), -0.5)
    return raw @ inverse_root
