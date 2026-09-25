from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core import stable_seed


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def nested_bootstrap(values: np.ndarray, replicates: int, seed: int) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("values must have shape training_blocks x target_graphs")
    blocks, graphs = values.shape
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        block_draws = rng.integers(0, blocks, size=(size, blocks))
        accumulator = np.zeros(size, dtype=np.float64)
        for position in range(blocks):
            selected = values[block_draws[:, position]]
            graph_draws = rng.integers(0, graphs, size=(size, graphs))
            accumulator += np.take_along_axis(selected, graph_draws, axis=1).mean(axis=1)
        result[start:start + size] = accumulator / blocks
    return result


def nested_shift_bootstrap(
    target: np.ndarray,
    source: np.ndarray,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if target.shape[0] != source.shape[0]:
        raise ValueError("target and source must have the same training blocks")
    blocks, target_graphs = target.shape
    source_graphs = source.shape[1]
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates - start)
        block_draws = rng.integers(0, blocks, size=(size, blocks))
        accumulator = np.zeros(size, dtype=np.float64)
        for position in range(blocks):
            chosen_target = target[block_draws[:, position]]
            chosen_source = source[block_draws[:, position]]
            target_draws = rng.integers(0, target_graphs, size=(size, target_graphs))
            source_draws = rng.integers(0, source_graphs, size=(size, source_graphs))
            target_mean = np.take_along_axis(chosen_target, target_draws, axis=1).mean(axis=1)
            source_mean = np.take_along_axis(chosen_source, source_draws, axis=1).mean(axis=1)
            accumulator += target_mean - source_mean
        result[start:start + size] = accumulator / blocks
    return result


def advantage_array(
    rows: list[dict[str, str]],
    *,
    condition: str,
    candidate: str,
    metric: str,
    block_seeds: list[int],
) -> np.ndarray:
    lookup = {
        (int(row["block_seed"]), row["architecture"], int(row["graph_index"])): float(row[metric])
        for row in rows if row["condition"] == condition
    }
    arrays = []
    for block_seed in block_seeds:
        graph_indices = sorted({
            graph_index for seed, architecture, graph_index in lookup
            if seed == block_seed and architecture == "mean"
        })
        block_values = []
        for graph_index in graph_indices:
            mean_value = lookup[(block_seed, "mean", graph_index)]
            candidate_value = lookup[(block_seed, candidate, graph_index)]
            if metric == "accuracy":
                block_values.append(candidate_value - mean_value)
            else:
                block_values.append(mean_value - candidate_value)
        arrays.append(block_values)
    result = np.asarray(arrays, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise RuntimeError(f"invalid paired metric array for {condition}/{candidate}/{metric}")
    return result


def interval(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    return tuple(float(value) for value in np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    results = Path(args.results)
    config = json.loads((results / "config.resolved.json").read_text())
    rows = read_csv(results / "graph_metrics.csv")
    block_seeds = [int(seed) for seed in config["training_blocks"]["seeds"]]
    candidates = [entry["name"] for entry in config["architectures"] if entry["name"] != "mean"]
    conditions = [entry["condition"] for entry in config["conditions"]]
    metrics = ["log_loss", "accuracy", "brier"]
    replicates = int(config["analysis"]["bootstrap_replicates"])
    alpha = float(config["analysis"]["ordinary_interval_alpha"])
    contrast_rows = []
    arrays: dict[tuple[str, str, str], np.ndarray] = {}
    samples: dict[tuple[str, str, str], np.ndarray] = {}
    for condition in conditions:
        for candidate in candidates:
            for metric in metrics:
                values = advantage_array(
                    rows,
                    condition=condition,
                    candidate=candidate,
                    metric=metric,
                    block_seeds=block_seeds,
                )
                key = (condition, candidate, metric)
                arrays[key] = values
                boot = nested_bootstrap(
                    values,
                    replicates,
                    stable_seed(config["analysis"]["bootstrap_seed"], "advantage", *key),
                )
                samples[key] = boot
                lower, upper = interval(boot, alpha)
                contrast_rows.append({
                    "condition": condition,
                    "candidate": candidate,
                    "metric": metric,
                    "estimand": "candidate_minus_mean" if metric == "accuracy" else "mean_minus_candidate",
                    "point": float(values.mean()),
                    "lower": lower,
                    "upper": upper,
                    "alpha": alpha,
                    "bootstrap_replicates": replicates,
                })
    write_csv(
        results / "condition_contrasts.csv",
        contrast_rows,
        ["condition", "candidate", "metric", "estimand", "point", "lower", "upper", "alpha", "bootstrap_replicates"],
    )

    secondary_rows = []
    if config["execution_mode"] == "full":
        secondary_alpha = float(config["analysis"]["secondary_two_sided_alpha_per_interval"])
        source_log_values = arrays[("source", "mean_degree", "log_loss")]
        for family, family_conditions in config["analysis"]["secondary_log_loss_families"].items():
            for condition in family_conditions:
                target_values = arrays[(condition, "mean_degree", "log_loss")]
                target_boot = samples[(condition, "mean_degree", "log_loss")]
                target_lower, target_upper = interval(target_boot, secondary_alpha)
                secondary_rows.append({
                    "family": family,
                    "condition": condition,
                    "estimand": "target_advantage",
                    "point": float(target_values.mean()),
                    "lower": target_lower,
                    "upper": target_upper,
                    "alpha": secondary_alpha,
                    "bootstrap_replicates": replicates,
                })
                gain_boot = nested_shift_bootstrap(
                    target_values,
                    source_log_values,
                    replicates,
                    stable_seed(config["analysis"]["bootstrap_seed"], "secondary_shift_gain", condition),
                )
                gain_lower, gain_upper = interval(gain_boot, secondary_alpha)
                secondary_rows.append({
                    "family": family,
                    "condition": condition,
                    "estimand": "shift_gain",
                    "point": float(target_values.mean() - source_log_values.mean()),
                    "lower": gain_lower,
                    "upper": gain_upper,
                    "alpha": secondary_alpha,
                    "bootstrap_replicates": replicates,
                })
    write_csv(
        results / "secondary_intervals.csv",
        secondary_rows,
        ["family", "condition", "estimand", "point", "lower", "upper", "alpha", "bootstrap_replicates"],
    )

    if config["execution_mode"] != "full":
        write_json(results / "primary_decision.json", {
            "status": "not_applicable_development_smoke",
            "scientific_evidence": False,
            "bootstrap_replicates": replicates,
        })
        write_csv(results / "primary_intervals.csv", [], [
            "condition", "metric", "estimand", "point", "lower", "upper", "alpha", "bootstrap_replicates",
        ])
        return

    primary_alpha = float(config["analysis"]["primary_one_sided_alpha_per_test"])
    primary_conditions = config["analysis"]["primary_conditions"]
    primary_rows = []
    decision_values: dict[str, dict[str, float]] = {}
    source_values = arrays[("source", "mean_degree", "log_loss")]
    for condition in primary_conditions:
        log_values = arrays[(condition, "mean_degree", "log_loss")]
        log_boot = samples[(condition, "mean_degree", "log_loss")]
        log_lower = float(np.quantile(log_boot, primary_alpha))
        log_upper = float(np.quantile(log_boot, 1.0 - primary_alpha))
        primary_rows.append({
            "condition": condition,
            "metric": "log_loss",
            "estimand": "target_advantage",
            "point": float(log_values.mean()),
            "lower": log_lower,
            "upper": log_upper,
            "alpha": primary_alpha,
            "bootstrap_replicates": replicates,
        })
        gain_boot = nested_shift_bootstrap(
            log_values,
            source_values,
            replicates,
            stable_seed(config["analysis"]["bootstrap_seed"], "shift_gain", condition),
        )
        gain_point = float(log_values.mean() - source_values.mean())
        gain_lower = float(np.quantile(gain_boot, primary_alpha))
        gain_upper = float(np.quantile(gain_boot, 1.0 - primary_alpha))
        primary_rows.append({
            "condition": condition,
            "metric": "log_loss",
            "estimand": "shift_gain",
            "point": gain_point,
            "lower": gain_lower,
            "upper": gain_upper,
            "alpha": primary_alpha,
            "bootstrap_replicates": replicates,
        })
        accuracy_boot = samples[(condition, "mean_degree", "accuracy")]
        accuracy_lower = float(np.quantile(accuracy_boot, primary_alpha))
        accuracy_upper = float(np.quantile(accuracy_boot, 1.0 - primary_alpha))
        primary_rows.append({
            "condition": condition,
            "metric": "accuracy",
            "estimand": "target_advantage",
            "point": float(arrays[(condition, "mean_degree", "accuracy")].mean()),
            "lower": accuracy_lower,
            "upper": accuracy_upper,
            "alpha": 2.0 * primary_alpha,
            "bootstrap_replicates": replicates,
        })
        decision_values[condition] = {
            "advantage_lower": log_lower,
            "advantage_upper": log_upper,
            "gain_lower": gain_lower,
            "gain_upper": gain_upper,
            "accuracy_lower": accuracy_lower,
            "accuracy_upper": accuracy_upper,
        }
    write_csv(
        results / "primary_intervals.csv",
        primary_rows,
        ["condition", "metric", "estimand", "point", "lower", "upper", "alpha", "bootstrap_replicates"],
    )
    target_margin = float(config["analysis"]["target_log_loss_advantage_margin"])
    gain_margin = float(config["analysis"]["shift_gain_margin"])
    accuracy_margin = float(config["analysis"]["accuracy_equivalence_margin"])
    little_margin = float(config["analysis"]["little_log_loss_margin"])
    transfer_benefit = all(value["advantage_lower"] > target_margin for value in decision_values.values())
    shift_amplified = transfer_benefit and any(value["gain_lower"] > gain_margin for value in decision_values.values())
    probability_only = transfer_benefit and all(
        value["accuracy_lower"] >= -accuracy_margin and value["accuracy_upper"] <= accuracy_margin
        for value in decision_values.values()
    )
    little_benefit = all(
        value["advantage_lower"] >= -little_margin and value["advantage_upper"] <= little_margin
        for value in decision_values.values()
    )
    harmful = any(value["advantage_upper"] < -little_margin for value in decision_values.values())
    if harmful:
        category = "harm_under_degree_shift"
    elif shift_amplified and probability_only:
        category = "shift_amplified_probability_only_transfer"
    elif shift_amplified:
        category = "shift_amplified_transfer"
    elif transfer_benefit and probability_only:
        category = "probability_only_transfer"
    elif transfer_benefit:
        category = "transfer_benefit_without_detected_amplification"
    elif little_benefit:
        category = "practically_little_benefit"
    else:
        category = "mixed_or_inconclusive"
    write_json(results / "primary_decision.json", {
        "status": "pending_validity_gates",
        "category_if_valid": category,
        "degree_improves_transfer": transfer_benefit,
        "shift_amplified_transfer": shift_amplified,
        "probability_only": probability_only,
        "practically_little_benefit": little_benefit,
        "harm_under_degree_shift": harmful,
        "conditions": decision_values,
        "margins": {
            "target_log_loss_advantage": target_margin,
            "shift_gain": gain_margin,
            "accuracy_equivalence": accuracy_margin,
            "little_log_loss": little_margin,
        },
        "bootstrap_replicates": replicates,
    })


if __name__ == "__main__":
    main()
