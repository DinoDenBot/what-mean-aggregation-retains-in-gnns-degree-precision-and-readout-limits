from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from core import (
    analytic_score_from_aggregate,
    canonical_contributions,
    canonical_direction_vectors,
    condition_quantities,
    generate_root_pool,
    orthogonal_fisher,
    raw_fisher_matrix,
)
from models import LocalAggregator


def check(name: str, condition: bool, details: dict | None = None) -> dict:
    if not condition:
        raise AssertionError(f"{name}: {details}")
    return {"name": name, "passed": True, "details": details or {}}


def explicit_adjacency_root_counts(
    repetitions: int,
    n_vertices: int,
    lambda0: float,
    rho0: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize complete symmetric CSBM adjacencies and return root counts."""
    values = condition_quantities(n_vertices, lambda0, rho0)
    labels = torch.cat([
        torch.ones(n_vertices // 2, dtype=torch.float64),
        -torch.ones(n_vertices // 2, dtype=torch.float64),
    ])
    probabilities = torch.empty((n_vertices, n_vertices), dtype=torch.float64)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    probabilities[same] = values["q_same"]
    probabilities[~same] = values["q_cross"]
    probabilities.fill_diagonal_(0.0)
    upper = torch.triu(torch.ones_like(probabilities, dtype=torch.bool), diagonal=1)
    generator = torch.Generator().manual_seed(seed)
    same_counts = []
    cross_counts = []
    for start in range(0, repetitions, 256):
        batch = min(256, repetitions - start)
        draws = torch.rand((batch, n_vertices, n_vertices), generator=generator, dtype=torch.float64)
        adjacency_upper = (draws < probabilities.unsqueeze(0)) & upper.unsqueeze(0)
        adjacency = adjacency_upper | adjacency_upper.transpose(1, 2)
        root_row = adjacency[:, 0]
        same_counts.append(root_row[:, 1:n_vertices // 2].sum(1))
        cross_counts.append(root_row[:, n_vertices // 2:].sum(1))
    return torch.cat(same_counts).to(torch.float64), torch.cat(cross_counts).to(torch.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).read_text())
    rows = []

    graph_size = 512
    expected_degree = 64.0
    rho0 = float(config["rho0"])
    mu0 = float(config["mu0"])
    quantities = condition_quantities(graph_size, expected_degree, rho0)
    expected = (
        quantities["same_candidates"] * quantities["q_same"]
        + quantities["cross_candidates"] * quantities["q_cross"]
    )
    rows.append(check("expected_degree_exact", abs(expected - expected_degree) < 1e-12, {"value": expected}))

    fisher = raw_fisher_matrix(graph_size, expected_degree, rho0)
    rows.append(check("raw_fisher_positive_definite", float(torch.linalg.eigvalsh(fisher).min()) > 0.0))

    pool = generate_root_pool(40000, graph_size, expected_degree, rho0, mu0, 1234, 0)
    covariance_error = float(torch.linalg.matrix_norm(torch.cov(pool.score.T) - torch.eye(3, dtype=torch.float64), ord=2))
    rows.append(check("monte_carlo_whitening", covariance_error < 0.035, {"operator_error": covariance_error}))

    for architecture in ("sum", "mean_degree"):
        model = LocalAggregator(architecture, expected_degree)
        aggregate = model.aggregate(pool.messages[:2048], pool.mask[:2048])
        recovered = analytic_score_from_aggregate(
            architecture, aggregate, graph_size, expected_degree, rho0, mu0
        )
        maximum_error = float(torch.abs(recovered - pool.score[:2048]).max())
        rows.append(check(f"{architecture}_score_recovery", maximum_error < 3e-5, {"maximum_error": maximum_error}))

    score = pool.score[:10000]
    estimate, per_matrix, _ = orthogonal_fisher(score, score)
    direct = score.T @ score / score.shape[0]
    rows.append(check("orthogonal_estimator_oracle_identity", float(torch.abs(estimate - direct).max()) < 1e-12))
    canonical = canonical_direction_vectors(fisher)
    contribution = canonical_contributions(per_matrix, canonical)
    rows.append(check("canonical_contribution_shape", tuple(contribution.shape) == (10000, 3)))

    messages = torch.tensor([[[1.0, 1.0, 0.2, 0.2], [1.0, -1.0, -0.4, 0.4]]])
    mask = torch.ones((1, 2), dtype=torch.bool)
    duplicated = messages.repeat_interleave(2, dim=1)
    duplicated_mask = mask.repeat_interleave(2, dim=1)
    for architecture in ("mean", "sum", "gat"):
        model = LocalAggregator(architecture, expected_degree)
        base = model.aggregate(messages, mask)
        repeated = model.aggregate(duplicated, duplicated_mask)
        if architecture in {"mean", "gat"}:
            condition = torch.allclose(base[:, :4], repeated[:, :4], atol=1e-6)
        else:
            condition = torch.allclose(2.0 * base[:, :4], repeated[:, :4], atol=1e-6)
        rows.append(check(f"multiplicity_behavior_{architecture}", bool(condition)))

    for architecture, expected_dimension in (("mean", 5), ("sum", 5), ("mean_degree", 6), ("gat", 5)):
        model = LocalAggregator(architecture, expected_degree)
        prediction, aggregate, hidden = model(pool.messages[:16], pool.mask[:16])
        rows.append(check(
            f"shape_{architecture}",
            tuple(prediction.shape) == (16, 3)
            and tuple(aggregate.shape) == (16, expected_dimension)
            and tuple(hidden.shape) == (16, 16),
        ))

    small_n = 32
    repetitions = 6000
    small_lambda = 6.0
    same, cross = explicit_adjacency_root_counts(repetitions, small_n, small_lambda, rho0, 9876)
    small = condition_quantities(small_n, small_lambda, rho0)
    theoretical = torch.tensor([
        small["same_candidates"] * small["q_same"],
        small["cross_candidates"] * small["q_cross"],
    ])
    empirical = torch.tensor([same.mean(), cross.mean()])
    variances = torch.tensor([
        small["same_candidates"] * small["q_same"] * (1.0 - small["q_same"]),
        small["cross_candidates"] * small["q_cross"] * (1.0 - small["q_cross"]),
    ])
    z = torch.abs(empirical - theoretical) / torch.sqrt(variances / repetitions)
    rows.append(check("explicit_adjacency_root_sampler_equivalence", bool(torch.all(z < 5.0)), {
        "standardized_mean_errors": z.tolist(),
    }))

    report = {"status": "passed", "tests": rows, "count": len(rows)}
    Path(arguments.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
