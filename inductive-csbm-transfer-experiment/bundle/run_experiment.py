from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from core import (
    GraphData,
    combine_graphs,
    degree_bin_label,
    generate_pool,
    set_reproducibility,
    source_degree_normalization,
    stable_seed,
    tensor_state_sha256,
    to_device,
)
from models import InductiveGNN, parameter_count


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def progress_writer(output: Path):
    path = output / "progress.jsonl"

    def emit(phase: str, completed: int, total: int, detail: dict | None = None) -> None:
        record = {
            "time": time.time(),
            "phase": phase,
            "completed": completed,
            "total": total,
            "detail": detail or {},
        }
        with path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)

    return emit


def graph_inventory_row(graph: GraphData, block_seed: int, pool: str) -> dict:
    lambda_, rho, mu = graph.environment
    mean_degree = float(graph.degree.to(torch.float64).mean())
    return {
        "block_seed": block_seed,
        "pool": pool,
        "condition": graph.condition,
        "graph_id": graph.graph_id,
        "n": graph.n,
        "undirected_edges": graph.undirected_edges,
        "mean_degree": mean_degree,
        "lambda": lambda_,
        "rho": rho,
        "mu": mu,
        "positive_labels": int(graph.y.sum()),
    }


@torch.no_grad()
def evaluation_loss(model: InductiveGNN, graph: GraphData, device: torch.device) -> float:
    model.eval()
    value = to_device(graph, device)
    logits = model(value.x, value.edge_src, value.degree)
    return float(functional.binary_cross_entropy_with_logits(logits, value.y).item())


def fit_model(
    model: InductiveGNN,
    train_graph: GraphData,
    validation_graph: GraphData,
    optimization: dict,
    seed: int,
    device: torch.device,
) -> tuple[InductiveGNN, dict]:
    set_reproducibility(seed)
    model.to(device)
    train = to_device(train_graph, device)
    validation = to_device(validation_graph, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    maximum_epochs = int(optimization["maximum_epochs"])
    minimum_epochs = int(optimization["minimum_epochs"])
    patience = int(optimization["early_stopping_patience"])
    min_improvement = float(optimization["early_stopping_minimum_improvement"])
    best_validation = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history = []
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train.x, train.edge_src, train.degree)
        loss = functional.binary_cross_entropy_with_logits(logits, train.y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss at epoch {epoch}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(optimization["gradient_clip_norm"]))
        if not torch.isfinite(norm):
            raise RuntimeError(f"nonfinite gradient norm at epoch {epoch}")
        optimizer.step()
        validation_loss = evaluation_loss(model, validation, device)
        history.append({
            "epoch": epoch,
            "training_log_loss": float(loss.item()),
            "validation_log_loss": validation_loss,
            "gradient_norm_before_clip": float(norm),
        })
        if validation_loss < best_validation - min_improvement:
            best_validation = validation_loss
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        if epoch >= minimum_epochs and stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, {
        "epochs": len(history),
        "best_validation_log_loss": best_validation,
        "best_state_sha256": tensor_state_sha256(best_state),
        "history": history,
    }


@torch.no_grad()
def graph_metrics(
    model: InductiveGNN,
    graph: GraphData,
    device: torch.device,
    degree_bins: list[list[int | None]],
    minimum_bin_nodes: int,
) -> tuple[dict, list[dict]]:
    model.eval()
    value = to_device(graph, device)
    logits = model(value.x, value.edge_src, value.degree)
    probabilities = torch.sigmoid(logits)
    node_log_loss = functional.binary_cross_entropy_with_logits(logits, value.y, reduction="none")
    node_accuracy = ((probabilities >= 0.5) == (value.y >= 0.5)).to(torch.float32)
    node_brier = torch.square(probabilities - value.y)
    overall = {
        "log_loss": float(node_log_loss.mean()),
        "accuracy": float(node_accuracy.mean()),
        "brier": float(node_brier.mean()),
    }
    rows = []
    for lower, upper in degree_bins:
        lower_value = int(lower)
        mask = value.degree >= lower_value
        if upper is not None:
            mask = mask & (value.degree <= int(upper))
        count = int(mask.sum())
        valid = count >= minimum_bin_nodes
        rows.append({
            "degree_bin": degree_bin_label(lower_value, int(upper) if upper is not None else None),
            "node_count": count,
            "valid": valid,
            "log_loss": float(node_log_loss[mask].mean()) if valid else "",
            "accuracy": float(node_accuracy[mask].mean()) if valid else "",
            "brier": float(node_brier[mask].mean()) if valid else "",
        })
    return overall, rows


def resolved_design(config: dict, mode: str) -> dict:
    design = copy.deepcopy(config)
    design["execution_mode"] = mode
    if mode == "full":
        return design
    design["protocol_status"] = "development_smoke_not_scientific"
    design["generator"]["n"] = 256
    design["training_blocks"]["seeds"] = [17]
    design["training_blocks"]["source_train_graphs"] = 2
    design["training_blocks"]["source_validation_graphs"] = 1
    design["training_blocks"]["target_graphs_per_condition"] = 2
    design["conditions"] = [
        condition for condition in design["conditions"]
        if condition["condition"] in {"source", "lambda_low"}
    ]
    design["optimization"]["maximum_epochs"] = 3
    design["optimization"]["minimum_epochs"] = 1
    design["optimization"]["early_stopping_patience"] = 3
    design["analysis"]["bootstrap_replicates"] = 20
    return design


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), required=True)
    parser.add_argument("--confirm-production")
    args = parser.parse_args()
    source_config = json.loads(Path(args.config).read_text())
    if args.mode == "full" and args.confirm_production != source_config["experiment_id"]:
        raise SystemExit("full mode requires --confirm-production inductive-csbm-transfer-v1")
    config = resolved_design(source_config, args.mode)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.resolved.json", config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(output / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pid": os.getpid(),
        "mode": args.mode,
    })
    progress = progress_writer(output)
    block_seeds = config["training_blocks"]["seeds"]
    architectures = config["architectures"]
    conditions = config["conditions"]
    total_training = len(block_seeds) * len(architectures)
    total_evaluation = len(block_seeds) * len(conditions) * int(config["training_blocks"]["target_graphs_per_condition"])
    total = total_training + total_evaluation
    completed = 0
    inventory_rows: list[dict] = []
    training_rows: list[dict] = []
    model_rows: list[dict] = []
    models: dict[tuple[int, str], InductiveGNN] = {}
    frozen_records: list[dict] = []
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    source_condition = config["source_environment"]

    # Phase 1: fit every model in every independent source block.
    for block_seed in block_seeds:
        train_graphs = generate_pool(
            config,
            block_seed=block_seed,
            pool="source_train",
            condition=source_condition,
            count=int(config["training_blocks"]["source_train_graphs"]),
        )
        validation_graphs = generate_pool(
            config,
            block_seed=block_seed,
            pool="source_validation",
            condition=source_condition,
            count=int(config["training_blocks"]["source_validation_graphs"]),
        )
        for graph in train_graphs:
            inventory_rows.append(graph_inventory_row(graph, block_seed, "source_train"))
        for graph in validation_graphs:
            inventory_rows.append(graph_inventory_row(graph, block_seed, "source_validation"))
        degree_location, degree_scale, pna_delta = source_degree_normalization(train_graphs)
        combined_train = combine_graphs(train_graphs, f"source_train_block:{block_seed}")
        combined_validation = combine_graphs(validation_graphs, f"source_validation_block:{block_seed}")
        for architecture in architectures:
            name = architecture["name"]
            model_seed = stable_seed(config["experiment_id"], "model", block_seed, name)
            set_reproducibility(model_seed)
            model = InductiveGNN(
                architecture=name,
                input_dimension=int(config["generator"]["feature_dimension"]),
                hidden_width=int(architecture["hidden_width"]),
                degree_location=degree_location,
                degree_scale=degree_scale,
                pna_delta=pna_delta,
                dropout=float(config["model"]["dropout"]),
            )
            count = parameter_count(model)
            model, information = fit_model(
                model,
                combined_train,
                combined_validation,
                config["optimization"],
                model_seed,
                device,
            )
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            state_hash = tensor_state_sha256(state)
            checkpoint = checkpoints / f"block-{block_seed}-{name}.pt"
            torch.save({"state_dict": state, "architecture": architecture}, checkpoint)
            models[(block_seed, name)] = model
            frozen_records.append({
                "block_seed": block_seed,
                "architecture": name,
                "state_sha256": state_hash,
                "checkpoint": checkpoint.relative_to(output).as_posix(),
                "degree_location": degree_location,
                "degree_scale": degree_scale,
                "pna_delta": pna_delta,
            })
            model_rows.append({
                "block_seed": block_seed,
                "architecture": name,
                "parameter_count": count,
                "hidden_width": architecture["hidden_width"],
                "epochs": information["epochs"],
                "best_validation_log_loss": information["best_validation_log_loss"],
                "state_sha256": state_hash,
            })
            for point in information["history"]:
                training_rows.append({"block_seed": block_seed, "architecture": name, **point})
            completed += 1
            progress("training", completed, total, {"block_seed": block_seed, "architecture": name})
        del combined_train, combined_validation
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_json(output / "WEIGHTS_FROZEN.json", {
        "experiment_id": config["experiment_id"],
        "created_before_target_generation": True,
        "models": frozen_records,
    })

    # Phase 2: generate independent target graphs and evaluate every frozen model.
    metric_rows: list[dict] = []
    bin_rows: list[dict] = []
    target_count = int(config["training_blocks"]["target_graphs_per_condition"])
    for block_seed in block_seeds:
        for condition in conditions:
            target_graphs = generate_pool(
                config,
                block_seed=block_seed,
                pool="target",
                condition=condition,
                count=target_count,
            )
            for graph_index, graph in enumerate(target_graphs):
                inventory_rows.append(graph_inventory_row(graph, block_seed, "target"))
                for architecture in architectures:
                    name = architecture["name"]
                    model = models[(block_seed, name)]
                    metrics, bins = graph_metrics(
                        model,
                        graph,
                        device,
                        config["evaluation"]["degree_bins"],
                        int(config["evaluation"]["minimum_nodes_per_graph_degree_bin"]),
                    )
                    common = {
                        "block_seed": block_seed,
                        "architecture": name,
                        "condition": condition["condition"],
                        "family": condition["family"],
                        "lambda": condition["lambda"],
                        "rho": condition["rho"],
                        "mu": condition["mu"],
                        "graph_index": graph_index,
                        "graph_id": graph.graph_id,
                    }
                    metric_rows.append({**common, **metrics})
                    bin_rows.extend([{**common, **row} for row in bins])
                completed += 1
                progress("target_evaluation", completed, total, {
                    "block_seed": block_seed,
                    "condition": condition["condition"],
                    "graph_index": graph_index,
                })

    post_hashes = []
    frozen_map = {(row["block_seed"], row["architecture"]): row["state_sha256"] for row in frozen_records}
    for key, model in models.items():
        current = tensor_state_sha256({name: value.detach().cpu() for name, value in model.state_dict().items()})
        post_hashes.append({
            "block_seed": key[0],
            "architecture": key[1],
            "before": frozen_map[key],
            "after": current,
            "unchanged": current == frozen_map[key],
        })
    write_json(output / "weights_post_evaluation.json", post_hashes)
    write_csv(
        output / "graph_inventory.csv", inventory_rows,
        ["block_seed", "pool", "condition", "graph_id", "n", "undirected_edges", "mean_degree", "lambda", "rho", "mu", "positive_labels"],
    )
    write_csv(
        output / "model_inventory.csv", model_rows,
        ["block_seed", "architecture", "parameter_count", "hidden_width", "epochs", "best_validation_log_loss", "state_sha256"],
    )
    write_csv(
        output / "training_diagnostics.csv", training_rows,
        ["block_seed", "architecture", "epoch", "training_log_loss", "validation_log_loss", "gradient_norm_before_clip"],
    )
    write_csv(
        output / "graph_metrics.csv", metric_rows,
        ["block_seed", "architecture", "condition", "family", "lambda", "rho", "mu", "graph_index", "graph_id", "log_loss", "accuracy", "brier"],
    )
    write_csv(
        output / "degree_bin_metrics.csv", bin_rows,
        ["block_seed", "architecture", "condition", "family", "lambda", "rho", "mu", "graph_index", "graph_id", "degree_bin", "node_count", "valid", "log_loss", "accuracy", "brier"],
    )
    progress("acquisition_complete", total, total, {"weights_unchanged": all(row["unchanged"] for row in post_hashes)})


if __name__ == "__main__":
    main()
