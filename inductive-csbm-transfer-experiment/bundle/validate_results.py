from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--unit-test-report", required=True)
    args = parser.parse_args()
    results = Path(args.results)
    config = json.loads((results / "config.resolved.json").read_text())
    unit_report = json.loads(Path(args.unit_test_report).read_text())
    inventory = read_csv(results / "graph_inventory.csv")
    models = read_csv(results / "model_inventory.csv")
    graph_metrics = read_csv(results / "graph_metrics.csv")
    degree_metrics = read_csv(results / "degree_bin_metrics.csv")
    post_hashes = json.loads((results / "weights_post_evaluation.json").read_text())
    progress = [json.loads(line) for line in (results / "progress.jsonl").read_text().splitlines() if line.strip()]
    decision = json.loads((results / "primary_decision.json").read_text())
    full = config["execution_mode"] == "full"
    blocks = len(config["training_blocks"]["seeds"])
    architectures = len(config["architectures"])
    conditions = len(config["conditions"])
    train_graphs = int(config["training_blocks"]["source_train_graphs"])
    validation_graphs = int(config["training_blocks"]["source_validation_graphs"])
    target_graphs = int(config["training_blocks"]["target_graphs_per_condition"])
    expected_inventory = blocks * (train_graphs + validation_graphs + conditions * target_graphs)
    expected_metrics = blocks * architectures * conditions * target_graphs
    expected_degree_metrics = expected_metrics * len(config["evaluation"]["degree_bins"])
    gates: list[dict] = []

    def gate(name: str, passed: bool, observed: object, threshold: object) -> None:
        gates.append({
            "gate": name,
            "passed": bool(passed),
            "observed": json.dumps(observed, sort_keys=True),
            "threshold": json.dumps(threshold, sort_keys=True),
        })

    gate("unit_tests", unit_report.get("status") == "passed", unit_report.get("status"), "passed")
    gate("graph_inventory_count", len(inventory) == expected_inventory, len(inventory), expected_inventory)
    gate("graph_metric_count", len(graph_metrics) == expected_metrics, len(graph_metrics), expected_metrics)
    gate("degree_bin_metric_count", len(degree_metrics) == expected_degree_metrics, len(degree_metrics), expected_degree_metrics)
    graph_ids = [row["graph_id"] for row in inventory]
    gate("graph_seed_namespaces_unique", len(graph_ids) == len(set(graph_ids)), len(set(graph_ids)), len(graph_ids))
    exact_balance = all(int(row["positive_labels"]) * 2 == int(row["n"]) for row in inventory)
    gate("exact_label_balance", exact_balance, exact_balance, True)
    maximum_degree_error = max(
        abs(float(row["mean_degree"]) - float(row["lambda"])) / float(row["lambda"])
        for row in inventory
    )
    degree_threshold = float(config["validity"]["realized_mean_degree_relative_error_max"])
    gate("realized_mean_degree", maximum_degree_error <= degree_threshold, maximum_degree_error, degree_threshold)

    values_finite = True
    metric_bounds = True
    for row in graph_metrics:
        values = [float(row[name]) for name in ("log_loss", "accuracy", "brier")]
        values_finite &= all(math.isfinite(value) for value in values)
        metric_bounds &= values[0] >= 0.0 and 0.0 <= values[1] <= 1.0 and 0.0 <= values[2] <= 1.0
    gate("finite_graph_metrics", values_finite, values_finite, True)
    gate("metric_bounds", metric_bounds, metric_bounds, True)

    paired: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    graph_id_by_key: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for row in graph_metrics:
        key = (row["block_seed"], row["condition"], row["graph_index"])
        paired[key].add(row["architecture"])
        graph_id_by_key[key].add(row["graph_id"])
    expected_architectures = {entry["name"] for entry in config["architectures"]}
    pairing_passed = all(value == expected_architectures for value in paired.values()) and all(
        len(value) == 1 for value in graph_id_by_key.values()
    )
    gate("matched_target_graphs", pairing_passed, len(paired), blocks * conditions * target_graphs)

    counts_by_block: dict[int, dict[str, int]] = defaultdict(dict)
    epochs_valid = True
    validations_finite = True
    for row in models:
        block = int(row["block_seed"])
        counts_by_block[block][row["architecture"]] = int(row["parameter_count"])
        epochs_valid &= int(row["epochs"]) >= int(config["optimization"]["minimum_epochs"])
        validations_finite &= math.isfinite(float(row["best_validation_log_loss"]))
    mean_degree_ratios = [values["mean_degree"] / values["mean"] for values in counts_by_block.values()]
    pna_ratios = [values["pna"] / values["mean"] for values in counts_by_block.values()]
    gate(
        "mean_degree_parameter_count",
        max(mean_degree_ratios) <= float(config["validity"]["mean_degree_parameter_ratio_max"]),
        max(mean_degree_ratios),
        config["validity"]["mean_degree_parameter_ratio_max"],
    )
    pna_lower = float(config["validity"]["pna_parameter_ratio_lower"])
    pna_upper = float(config["validity"]["pna_parameter_ratio_upper"])
    gate("pna_parameter_count", min(pna_ratios) >= pna_lower and max(pna_ratios) <= pna_upper, [min(pna_ratios), max(pna_ratios)], [pna_lower, pna_upper])
    gate("minimum_training_epochs", epochs_valid, epochs_valid, True)
    gate("finite_validation_checkpoints", validations_finite, validations_finite, True)
    weights_unchanged = all(bool(row["unchanged"]) and row["before"] == row["after"] for row in post_hashes)
    gate("weights_unchanged_after_target_evaluation", weights_unchanged, weights_unchanged, True)
    training_positions = [index for index, row in enumerate(progress) if row["phase"] == "training"]
    target_positions = [index for index, row in enumerate(progress) if row["phase"] == "target_evaluation"]
    phase_order = bool(training_positions and target_positions and max(training_positions) < min(target_positions))
    phase_order &= (results / "WEIGHTS_FROZEN.json").exists()
    gate("global_freeze_before_target_evaluation", phase_order, phase_order, True)
    configured_bootstrap = int(config["analysis"]["bootstrap_replicates"])
    gate("bootstrap_replicates", configured_bootstrap == (10000 if full else 20), configured_bootstrap, 10000 if full else 20)

    all_valid = all(row["passed"] for row in gates)
    if full:
        decision["status"] = "valid" if all_valid else "invalid"
        decision["scientific_status"] = decision.get("category_if_valid") if all_valid else "invalid_due_to_failed_validity_gate"
    else:
        decision["status"] = "development_smoke_valid" if all_valid else "development_smoke_invalid"
        decision["scientific_status"] = "not_applicable"
    write_json(results / "primary_decision.json", decision)
    write_csv(results / "validity_gates.csv", gates, ["gate", "passed", "observed", "threshold"])
    summary = {
        "experiment_id": config["experiment_id"],
        "execution_mode": config["execution_mode"],
        "valid": all_valid,
        "scientific_status": decision["scientific_status"],
        "validity_gates_passed": sum(row["passed"] for row in gates),
        "validity_gates_total": len(gates),
        "training_blocks": blocks,
        "architectures": architectures,
        "conditions": conditions,
        "target_graphs_per_condition_per_block": target_graphs,
        "graph_metric_rows": len(graph_metrics),
        "maximum_realized_mean_degree_relative_error": maximum_degree_error,
    }
    write_json(results / "metrics.json", summary)
    write_json(results / "result_validation.json", {
        "status": "passed" if all_valid else "failed",
        "failed_gates": [row["gate"] for row in gates if not row["passed"]],
        "summary": summary,
    })
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all_valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

