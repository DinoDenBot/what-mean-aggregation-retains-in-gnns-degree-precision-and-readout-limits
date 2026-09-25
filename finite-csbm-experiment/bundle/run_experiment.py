from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from core import (
    DIRECTIONS,
    analytic_score_from_aggregate,
    canonical_contributions,
    canonical_direction_vectors,
    encode_pool,
    fit_architecture,
    fit_nuisance,
    generate_root_pool,
    orthogonal_fisher,
    predict_nuisance,
    raw_fisher_matrix,
    stable_seed,
)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def progress_writer(output: Path):
    path = output / "progress.jsonl"

    def emit(phase: str, completed: int, total: int, detail: dict | None = None) -> None:
        row = {"time": time.time(), "phase": phase, "completed": completed, "total": total, "detail": detail or {}}
        with path.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)

    return emit


def matrix_rows(matrix: torch.Tensor, graph_size: int, expected_degree: int, seed: int, architecture: str, observation: str) -> list[dict]:
    return [
        {
            "graph_size": graph_size,
            "expected_degree": expected_degree,
            "seed": seed,
            "architecture": architecture,
            "observation": observation,
            "row": row,
            "column": column,
            "value": float(matrix[row, column]),
        }
        for row in range(3)
        for column in range(3)
    ]


def diagnostics(matrix: torch.Tensor) -> tuple[list[float], float]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    return [float(value) for value in eigenvalues], float(eigenvalues.min())


def bootstrap_condition(
    contributions: dict[tuple[str, str], torch.Tensor],
    architectures: list[str],
    graph_size: int,
    expected_degree: int,
    seed: int,
    replicates: int,
    device: torch.device,
) -> tuple[list[dict], list[dict]]:
    keys = [(name, observation) for name in architectures for observation in ("A", "H")]
    stacked = torch.stack([contributions[key] for key in keys], dim=1).to(torch.float32)
    count = stacked.shape[0]
    samples = []
    generator = torch.Generator(device=device).manual_seed(
        stable_seed("finite-csbm-bootstrap", graph_size, expected_degree, seed)
    )
    data = stacked.to(device)
    for start in range(0, replicates, 100):
        current = min(100, replicates - start)
        indices = torch.randint(0, count, (current, count), generator=generator, device=device)
        samples.append(data[indices].mean(1).cpu())
    draws = torch.cat(samples).to(torch.float64)
    intervals = []
    for key_index, (architecture, observation) in enumerate(keys):
        for direction_index, direction in enumerate(DIRECTIONS):
            values = draws[:, key_index, direction_index]
            lower, upper = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
            intervals.append({
                "graph_size": graph_size,
                "expected_degree": expected_degree,
                "seed": seed,
                "architecture": architecture,
                "observation": observation,
                "direction": direction,
                "point": float(stacked[:, key_index, direction_index].mean()),
                "lower": lower,
                "upper": upper,
                "replicates": replicates,
            })
    lookup = {key: index for index, key in enumerate(keys)}
    contrasts = []
    for label, left, right in (
        ("sum_minus_mean", "sum", "mean"),
        ("mean_degree_minus_mean", "mean_degree", "mean"),
    ):
        for direction_index, direction in enumerate(DIRECTIONS):
            values = draws[:, lookup[(left, "A")], direction_index] - draws[:, lookup[(right, "A")], direction_index]
            lower, upper = torch.quantile(values, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
            contrasts.append({
                "graph_size": graph_size,
                "expected_degree": expected_degree,
                "seed": seed,
                "contrast": label,
                "direction": direction,
                "point": float(values.mean()),
                "lower": lower,
                "upper": upper,
                "replicates": replicates,
            })
    return intervals, contrasts


def run(config: dict, output: Path, mode: str, device: torch.device, progress) -> dict:
    if mode == "full":
        graph_sizes = config["graph_sizes"]
        expected_degrees = config["expected_degrees"]
        seeds = config["seeds"]
        architectures = [row["name"] for row in config["architectures"]]
        pool_sizes = config["independent_pools_per_condition_seed"]
        architecture_steps = None
        nuisance_steps = None
        bootstrap_replicates = int(config["bootstrap"]["replicates"])
    else:
        graph_sizes = [512]
        expected_degrees = [16]
        seeds = [17]
        architectures = ["mean", "sum", "mean_degree", "gat"]
        pool_sizes = {
            "architecture_train": 256,
            "architecture_validation": 128,
            "nuisance_fit": 256,
            "nuisance_validation": 128,
            "fisher_evaluation": 256,
        }
        architecture_steps = 5
        nuisance_steps = 5
        bootstrap_replicates = 20

    rho0 = float(config["rho0"])
    mu0 = float(config["mu0"])
    directional_rows = []
    spectrum_rows = []
    fisher_rows = []
    interval_rows = []
    contrast_rows = []
    gap_rows = []
    raw_rows = []
    oracle_rows = []
    learned_oracle_rows = []
    nuisance_rows = []
    training_rows = []
    pool_rows = []
    total = len(graph_sizes) * len(expected_degrees) * len(seeds) * len(architectures)
    completed = 0

    for graph_index, graph_size in enumerate(graph_sizes):
        for degree_index, expected_degree in enumerate(expected_degrees):
            fisher = raw_fisher_matrix(graph_size, expected_degree, rho0)
            canonical = canonical_direction_vectors(fisher)
            for seed_index, seed in enumerate(seeds):
                base = (graph_index * 100 + degree_index * 10 + seed_index) * 10_000_000_000
                specifications = (
                    ("architecture_train", 0),
                    ("architecture_validation", 1),
                    ("nuisance_fit", 2),
                    ("nuisance_validation", 3),
                    ("fisher_evaluation", 4),
                )
                pools = {}
                for pool_name, pool_index in specifications:
                    pools[pool_name] = generate_root_pool(
                        int(pool_sizes[pool_name]),
                        graph_size,
                        expected_degree,
                        rho0,
                        mu0,
                        stable_seed("finite-csbm", graph_size, expected_degree, seed, pool_name),
                        base + pool_index * 1_000_000_000,
                    )
                    pool = pools[pool_name]
                    pool_rows.append({
                        "graph_size": graph_size,
                        "expected_degree": expected_degree,
                        "seed": seed,
                        "pool": pool_name,
                        "minimum_id": int(pool.sample_ids.min()),
                        "maximum_id": int(pool.sample_ids.max()),
                    })
                ranges = [(row["minimum_id"], row["maximum_id"]) for row in pool_rows[-5:]]
                for left in range(len(ranges)):
                    for right in range(left + 1, len(ranges)):
                        if not (ranges[left][1] < ranges[right][0] or ranges[right][1] < ranges[left][0]):
                            raise RuntimeError("sample-ID overlap")
                raw_covariance = torch.cov(pools["fisher_evaluation"].score.T)
                raw_rows.append({
                    "graph_size": graph_size,
                    "expected_degree": expected_degree,
                    "seed": seed,
                    "operator_error": float(torch.linalg.matrix_norm(raw_covariance - torch.eye(3, dtype=torch.float64), ord=2)),
                })
                contribution_map = {}
                condition_estimates = {}
                condition_oracles = {}
                for architecture in architectures:
                    model, training = fit_architecture(
                        architecture,
                        expected_degree,
                        pools["architecture_train"],
                        pools["architecture_validation"],
                        config,
                        stable_seed("finite-csbm-architecture", graph_size, expected_degree, seed, architecture),
                        device,
                        architecture_steps,
                    )
                    for point in training["curve"]:
                        training_rows.append({
                            "graph_size": graph_size,
                            "expected_degree": expected_degree,
                            "seed": seed,
                            "architecture": architecture,
                            "stage": "architecture",
                            **point,
                        })
                    encoded = {
                        pool_name: encode_pool(model, pools[pool_name], device)
                        for pool_name in ("nuisance_fit", "nuisance_validation", "fisher_evaluation")
                    }
                    estimates = {}
                    for observation, index in (("A", 0), ("H", 1)):
                        nuisance, nuisance_fit = fit_nuisance(
                            encoded["nuisance_fit"][index],
                            pools["nuisance_fit"].score,
                            encoded["nuisance_validation"][index],
                            pools["nuisance_validation"].score,
                            config,
                            stable_seed("finite-csbm-nuisance", graph_size, expected_degree, seed, architecture, observation),
                            device,
                            nuisance_steps,
                        )
                        for point in nuisance_fit["curve"]:
                            training_rows.append({
                                "graph_size": graph_size,
                                "expected_degree": expected_degree,
                                "seed": seed,
                                "architecture": architecture,
                                "stage": f"nuisance_{observation}",
                                **point,
                            })
                        prediction = predict_nuisance(nuisance, encoded["fisher_evaluation"][index], device)
                        estimate, per_matrix, residual = orthogonal_fisher(prediction, pools["fisher_evaluation"].score)
                        estimates[observation] = estimate
                        condition_estimates[(architecture, observation)] = estimate
                        fisher_rows.extend(matrix_rows(estimate, graph_size, expected_degree, seed, architecture, observation))
                        canonical_per_sample = canonical_contributions(per_matrix, canonical)
                        contribution_map[(architecture, observation)] = canonical_per_sample.to(torch.float32)
                        canonical_point = canonical_contributions(estimate.unsqueeze(0), canonical)[0]
                        for direction, value in zip(DIRECTIONS, canonical_point.tolist()):
                            directional_rows.append({
                                "graph_size": graph_size,
                                "expected_degree": expected_degree,
                                "seed": seed,
                                "architecture": architecture,
                                "observation": observation,
                                "direction": direction,
                                "efficiency": value,
                            })
                        eigenvalues, _ = diagnostics(estimate)
                        for rank, value in enumerate(eigenvalues, start=1):
                            spectrum_rows.append({
                                "graph_size": graph_size,
                                "expected_degree": expected_degree,
                                "seed": seed,
                                "architecture": architecture,
                                "observation": observation,
                                "rank": rank,
                                "eigenvalue": value,
                            })
                        nuisance_rows.append({
                            "graph_size": graph_size,
                            "expected_degree": expected_degree,
                            "seed": seed,
                            "architecture": architecture,
                            "observation": observation,
                            "residual_mse": residual,
                            "best_validation_mse": nuisance_fit["best_validation_mse"],
                        })

                    aggregate = encoded["fisher_evaluation"][0]
                    oracle_score = analytic_score_from_aggregate(
                        architecture, aggregate, graph_size, expected_degree, rho0, mu0
                    )
                    if oracle_score is not None:
                        oracle_matrix = oracle_score.T @ oracle_score / oracle_score.shape[0]
                        condition_oracles[architecture] = oracle_matrix
                        oracle_point = canonical_contributions(oracle_matrix.unsqueeze(0), canonical)[0]
                        learned_point = canonical_contributions(estimates["A"].unsqueeze(0), canonical)[0]
                        for direction, oracle_value, learned_value in zip(
                            DIRECTIONS, oracle_point.tolist(), learned_point.tolist()
                        ):
                            oracle_rows.append({
                                "graph_size": graph_size,
                                "expected_degree": expected_degree,
                                "seed": seed,
                                "architecture": architecture,
                                "direction": direction,
                                "efficiency": oracle_value,
                            })
                            learned_oracle_rows.append({
                                "graph_size": graph_size,
                                "expected_degree": expected_degree,
                                "seed": seed,
                                "architecture": architecture,
                                "direction": direction,
                                "learned_efficiency": learned_value,
                                "oracle_efficiency": oracle_value,
                                "absolute_error": abs(learned_value - oracle_value),
                            })
                    gap = 0.5 * ((estimates["A"] - estimates["H"]) + (estimates["A"] - estimates["H"]).T)
                    gap_rows.append({
                        "graph_size": graph_size,
                        "expected_degree": expected_degree,
                        "seed": seed,
                        "architecture": architecture,
                        "trace_gap": float(torch.trace(gap)),
                        "maximum_violation_eigenvalue": float(torch.linalg.eigvalsh(-gap).max()),
                    })
                    completed += 1
                    progress("architecture", completed, total, {
                        "graph_size": graph_size,
                        "expected_degree": expected_degree,
                        "seed": seed,
                        "architecture": architecture,
                    })
                intervals, contrasts = bootstrap_condition(
                    contribution_map,
                    architectures,
                    graph_size,
                    expected_degree,
                    seed,
                    bootstrap_replicates,
                    device,
                )
                interval_rows.extend(intervals)
                contrast_rows.extend(contrasts)

    return {
        "directional": directional_rows,
        "spectra": spectrum_rows,
        "fisher": fisher_rows,
        "intervals": interval_rows,
        "contrasts": contrast_rows,
        "gaps": gap_rows,
        "raw": raw_rows,
        "oracles": oracle_rows,
        "learned_oracles": learned_oracle_rows,
        "nuisance": nuisance_rows,
        "training": training_rows,
        "pools": pool_rows,
    }


def apply_decision(config: dict, results: dict, unit_ok: bool) -> dict:
    primary = config["primary_condition"]
    graph_size = primary["graph_size"]
    expected_degree = primary["expected_degree"]
    validity = config["validity_thresholds"]
    thresholds = config["decision_thresholds"]
    seeds = config["seeds"]
    raw_ok = all(row["operator_error"] <= validity["raw_score_covariance_operator_error_max"] for row in results["raw"])
    oracle_ok = all(
        abs(row["efficiency"] - 1.0) <= validity["analytic_sufficient_oracle_directional_error_max"]
        for row in results["oracles"]
    )
    learned_ok = all(
        row["absolute_error"] <= validity["learned_sufficient_oracle_directional_error_max"]
        for row in results["learned_oracles"]
    )
    minimum_ok = all(
        row["eigenvalue"] >= validity["unprojected_minimum_eigenvalue_min"]
        for row in results["spectra"]
        if row["graph_size"] == graph_size and row["expected_degree"] == expected_degree
    )
    decomposition_ok = all(
        row["maximum_violation_eigenvalue"] <= validity["estimated_data_processing_violation_eigenvalue_max"]
        for row in results["gaps"]
        if row["graph_size"] == graph_size and row["expected_degree"] == expected_degree
    )
    pool_ok = len(results["pools"]) > 0
    gate = unit_ok and raw_ok and oracle_ok and learned_ok and minimum_ok and decomposition_ok and pool_ok

    def interval(seed: int, architecture: str, direction: str) -> dict:
        return next(
            row for row in results["intervals"]
            if row["graph_size"] == graph_size
            and row["expected_degree"] == expected_degree
            and row["seed"] == seed
            and row["architecture"] == architecture
            and row["observation"] == "A"
            and row["direction"] == direction
        )

    def contrast(seed: int, label: str) -> dict:
        return next(
            row for row in results["contrasts"]
            if row["graph_size"] == graph_size
            and row["expected_degree"] == expected_degree
            and row["seed"] == seed
            and row["contrast"] == label
            and row["direction"] == "rate"
        )

    mean_support = gate and all(
        interval(seed, "mean", "rate")["upper"] < thresholds["mean_rate_upper_max"]
        and interval(seed, "mean", "homophily")["lower"] > thresholds["mean_homophily_and_context_lower_min"]
        and interval(seed, "mean", "context")["lower"] > thresholds["mean_homophily_and_context_lower_min"]
        for seed in seeds
    )
    restoration_support = gate and all(
        interval(seed, architecture, direction)["lower"] > thresholds["sufficient_control_all_directions_lower_min"]
        for seed in seeds
        for architecture in ("sum", "mean_degree")
        for direction in DIRECTIONS
    ) and all(
        contrast(seed, label)["lower"] > thresholds["sufficient_control_rate_advantage_lower_min"]
        for seed in seeds
        for label in ("sum_minus_mean", "mean_degree_minus_mean")
    )
    contradiction = gate and (
        all(interval(seed, "mean", "rate")["lower"] > thresholds["contradiction_mean_rate_lower_min"] for seed in seeds)
        or any(
            all(interval(seed, architecture, "rate")["upper"] < thresholds["contradiction_sufficient_control_rate_upper_max"] for seed in seeds)
            for architecture in ("sum", "mean_degree")
        )
    )
    verdict = "supported" if mean_support and restoration_support else ("contradicted" if contradiction else "inconclusive")
    return {
        "verdict": verdict,
        "validity": {
            "unit_tests": unit_ok,
            "raw_score_covariance": raw_ok,
            "analytic_sufficient_oracles": oracle_ok,
            "learned_sufficient_oracles": learned_ok,
            "minimum_eigenvalues": minimum_ok,
            "aggregation_to_trunk_decomposition": decomposition_ok,
            "sample_pool_independence": pool_ok,
            "all_primary_gates": gate,
        },
        "confirmatory": {
            "mean_rate_homophily_context_separation": mean_support,
            "sufficient_control_restoration": restoration_support,
            "contradiction": contradiction,
        },
    }


def make_figures(output: Path, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["directional"]
    architectures = ["mean", "gat", "sum", "mean_degree"]
    conditions = sorted({(row["graph_size"], row["expected_degree"]) for row in rows})
    figure, axes = plt.subplots(1, len(conditions), figsize=(3.2 * len(conditions), 4.3), constrained_layout=True)
    for axis, (graph_size, expected_degree) in zip(np.atleast_1d(axes), conditions):
        data = np.zeros((len(architectures), 3))
        for architecture_index, architecture in enumerate(architectures):
            for direction_index, direction in enumerate(DIRECTIONS):
                values = [
                    row["efficiency"] for row in rows
                    if row["graph_size"] == graph_size
                    and row["expected_degree"] == expected_degree
                    and row["architecture"] == architecture
                    and row["observation"] == "A"
                    and row["direction"] == direction
                ]
                data[architecture_index, direction_index] = np.median(values)
        image = axis.imshow(data, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_title(f"n={graph_size}, degree={expected_degree}")
        axis.set_xticks(range(3), ["rate", "hom.", "context"])
        axis.set_yticks(range(len(architectures)), architectures)
        for row_index in range(len(architectures)):
            for column_index in range(3):
                value = data[row_index, column_index]
                axis.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.55 else "black", fontsize=7)
    figure.colorbar(image, ax=np.atleast_1d(axes).tolist(), shrink=0.8, label="directional Fisher efficiency")
    figure.savefig(output / "figures" / "finite_csbm_heatmap.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    arguments = parser.parse_args()
    start = time.time()
    config_path = Path(arguments.config).resolve()
    output = Path(arguments.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    config = json.loads(config_path.read_text())
    shutil.copyfile(config_path, output / "config.resolved.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mode": arguments.mode,
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    progress = progress_writer(output)
    progress("start", 0, 1, environment)
    results = run(config, output, arguments.mode, device, progress)
    write_csv(output / "directional_efficiency.csv", results["directional"], ["graph_size", "expected_degree", "seed", "architecture", "observation", "direction", "efficiency"])
    write_csv(output / "spectra.csv", results["spectra"], ["graph_size", "expected_degree", "seed", "architecture", "observation", "rank", "eigenvalue"])
    write_csv(output / "fisher_matrices.csv", results["fisher"], ["graph_size", "expected_degree", "seed", "architecture", "observation", "row", "column", "value"])
    write_csv(output / "bootstrap_intervals.csv", results["intervals"], ["graph_size", "expected_degree", "seed", "architecture", "observation", "direction", "point", "lower", "upper", "replicates"])
    write_csv(output / "paired_contrasts.csv", results["contrasts"], ["graph_size", "expected_degree", "seed", "contrast", "direction", "point", "lower", "upper", "replicates"])
    write_csv(output / "aggregation_trunk_gap.csv", results["gaps"], ["graph_size", "expected_degree", "seed", "architecture", "trace_gap", "maximum_violation_eigenvalue"])
    write_csv(output / "raw_score_checks.csv", results["raw"], ["graph_size", "expected_degree", "seed", "operator_error"])
    write_csv(output / "oracle_checks.csv", results["oracles"], ["graph_size", "expected_degree", "seed", "architecture", "direction", "efficiency"])
    write_csv(output / "learned_oracle_checks.csv", results["learned_oracles"], ["graph_size", "expected_degree", "seed", "architecture", "direction", "learned_efficiency", "oracle_efficiency", "absolute_error"])
    write_csv(output / "nuisance_residuals.csv", results["nuisance"], ["graph_size", "expected_degree", "seed", "architecture", "observation", "residual_mse", "best_validation_mse"])
    write_csv(output / "training_diagnostics.csv", results["training"], ["graph_size", "expected_degree", "seed", "architecture", "stage", "step", "train_mse", "validation_mse"])
    write_csv(output / "pool_id_ranges.csv", results["pools"], ["graph_size", "expected_degree", "seed", "pool", "minimum_id", "maximum_id"])
    unit_path = output / "unit_test_report.json"
    unit_ok = unit_path.exists() and json.loads(unit_path.read_text()).get("status") == "passed"
    decision = apply_decision(config, results, unit_ok) if arguments.mode == "full" else {"verdict": "smoke_only", "validity": {"unit_tests": unit_ok}}
    make_figures(output, results)
    metrics = {
        "status": "completed",
        "mode": arguments.mode,
        "decision": decision,
        "runtime_seconds": time.time() - start,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "counts": {
            "directional_rows": len(results["directional"]),
            "bootstrap_rows": len(results["intervals"]),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    progress("complete", 1, 1, {"verdict": decision["verdict"], "runtime_seconds": metrics["runtime_seconds"]})


if __name__ == "__main__":
    main()
