from __future__ import annotations

import argparse
import inspect
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from analyze_results import nested_bootstrap, nested_shift_bootstrap
from core import (
    combine_graphs,
    edge_probabilities,
    generate_graph,
    graph_identifier,
    sample_bipartite_gnp,
    sample_undirected_gnp,
    stable_seed,
    tensor_state_sha256,
)
from models import InductiveGNN, parameter_count


def close(left: torch.Tensor, right: torch.Tensor, tolerance: float = 1e-6) -> None:
    if not torch.allclose(left, right, atol=tolerance, rtol=tolerance):
        raise AssertionError(f"maximum difference {torch.max(torch.abs(left-right))} exceeds {tolerance}")


def test_config(config: dict) -> None:
    assert config["experiment_id"] == "inductive-csbm-transfer-v1"
    assert config["protocol_status"] == "preregistered_not_run"
    assert config["training_blocks"]["seeds"] == [17, 29, 43, 71, 101, 131, 173, 211]
    assert config["analysis"]["bootstrap_replicates"] == 10000
    assert config["analysis"]["primary_conditions"] == ["lambda_low", "lambda_high"]
    assert config["analysis"]["secondary_log_loss_families"] == {
        "homophily": ["rho_low", "rho_high"],
        "feature_signal": ["mu_low", "mu_high"],
    }
    assert len(config["conditions"]) == 15
    assert {entry["name"] for entry in config["architectures"]} == {"mean", "mean_degree", "sum", "pna"}
    environments = {(entry["condition"], entry["lambda"], entry["rho"], entry["mu"]) for entry in config["conditions"]}
    assert len(environments) == 15


def test_environment_versions() -> None:
    assert torch.__version__.split("+")[0] == "2.8.0"
    assert np.__version__ == "2.1.2"


def test_seed_namespaces() -> None:
    values = set()
    for block in (17, 29):
        for pool in ("source_train", "source_validation", "target"):
            for condition in ("source", "lambda_low"):
                for graph_index in range(3):
                    identifier = graph_identifier("inductive-csbm-transfer-v1", block, pool, condition, graph_index)
                    assert identifier not in values
                    values.add(identifier)
    assert stable_seed("a", 1) == stable_seed("a", 1)
    assert stable_seed("a", 1) != stable_seed("a", 2)


def test_triangle_sampler_all_edges_and_simple() -> None:
    rng = np.random.default_rng(17)
    left, right = sample_undirected_gnp(8, 1.0, rng)
    assert len(left) == 28
    pairs = set(zip(left.tolist(), right.tolist()))
    assert len(pairs) == 28
    assert all(0 <= b < a < 8 for a, b in pairs)
    expected = {(a, b) for a in range(1, 8) for b in range(a)}
    assert pairs == expected


def test_edge_sampler_frequencies() -> None:
    repetitions = 4000
    probability = 0.2
    within_counts = []
    cross_counts = []
    rng = np.random.default_rng(29)
    for _ in range(repetitions):
        within_counts.append(len(sample_undirected_gnp(8, probability, rng)[0]))
        cross_counts.append(len(sample_bipartite_gnp(5, 7, probability, rng)[0]))
    within_expected = 28 * probability
    cross_expected = 35 * probability
    within_se = math.sqrt(28 * probability * (1 - probability) / repetitions)
    cross_se = math.sqrt(35 * probability * (1 - probability) / repetitions)
    assert abs(np.mean(within_counts) - within_expected) < 6 * within_se
    assert abs(np.mean(cross_counts) - cross_expected) < 6 * cross_se


def test_environment_expected_degree() -> None:
    n = 4096
    for lambda_, rho in ((12.0, 0.3), (24.0, 0.6), (48.0, 0.8)):
        same, cross = edge_probabilities(n, lambda_, rho)
        expected = (n // 2 - 1) * same + (n // 2) * cross
        assert abs(expected - lambda_) < 1e-12


def test_graph_reproducibility_balance_and_symmetry() -> None:
    arguments = dict(
        n=256,
        feature_dimension=8,
        lambda_=12.0,
        rho=0.6,
        mu=0.35,
        seed=stable_seed("test-graph"),
        graph_id="test",
        condition="source",
    )
    left = generate_graph(**arguments)
    right = generate_graph(**arguments)
    close(left.x, right.x, 0.0)
    assert torch.equal(left.y, right.y)
    assert torch.equal(left.edge_src, right.edge_src)
    assert torch.equal(left.edge_dst, right.edge_dst)
    assert int(left.y.sum()) == 128
    assert torch.equal(torch.bincount(left.edge_dst, minlength=left.n), left.degree)
    directed = set(zip(left.edge_src.tolist(), left.edge_dst.tolist()))
    assert len(directed) == left.edge_src.numel()
    assert all(source != target for source, target in directed)
    assert all((target, source) in directed for source, target in directed)
    assert bool(torch.all(left.edge_dst[:-1] <= left.edge_dst[1:]))


def tiny_graph() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Undirected edges: 0--1, 0--2, 2--3, stored in destination-major order.
    source = torch.tensor([1, 2, 0, 0, 3, 2], dtype=torch.int64)
    destination = torch.tensor([0, 0, 1, 2, 2, 3], dtype=torch.int64)
    degree = torch.bincount(destination, minlength=4)
    x = torch.tensor([[1.0, 0.0], [2.0, 1.0], [4.0, 2.0], [8.0, 3.0]])
    return x, source, degree


def make_model(name: str, hidden: int = 8) -> InductiveGNN:
    return InductiveGNN(name, 2, hidden, 2.0, 0.5, 2.0, 0.0).eval()


def test_aggregator_identities_and_permutation() -> None:
    x, source, degree = tiny_graph()
    mean_model = make_model("mean")
    sum_model = make_model("sum")
    degree_model = make_model("mean_degree")
    mean = mean_model.layer1.aggregate(x, source, degree)
    summed = sum_model.layer1.aggregate(x, source, degree)
    degree_augmented = degree_model.layer1.aggregate(x, source, degree)
    close(mean[0], torch.tensor([3.0, 1.5]))
    close(summed[2], torch.tensor([9.0, 3.0]))
    close(degree_augmented[:, :2], mean)
    expected_degree = (torch.log1p(degree.to(torch.float32)) - 2.0) / 0.5
    close(degree_augmented[:, -1], expected_degree)
    # Reorder messages within the two destination segments of length two.
    permuted = source.clone()
    permuted[:2] = permuted[:2].flip(0)
    permuted[3:5] = permuted[3:5].flip(0)
    for name in ("mean", "mean_degree", "sum", "pna"):
        model = make_model(name)
        close(model.layer1.aggregate(x, source, degree), model.layer1.aggregate(x, permuted, degree), 2e-6)


def test_model_interface_excludes_labels() -> None:
    parameters = list(inspect.signature(InductiveGNN.forward).parameters)
    assert parameters == ["self", "x", "edge_src", "degree"]
    source = inspect.getsource(InductiveGNN.forward)
    assert "label" not in source and ".y" not in source


def test_parameter_budgets() -> None:
    counts = {
        "mean": parameter_count(InductiveGNN("mean", 8, 64, 3.0, 0.5, 3.0, 0.1)),
        "mean_degree": parameter_count(InductiveGNN("mean_degree", 8, 64, 3.0, 0.5, 3.0, 0.1)),
        "sum": parameter_count(InductiveGNN("sum", 8, 64, 3.0, 0.5, 3.0, 0.1)),
        "pna": parameter_count(InductiveGNN("pna", 8, 24, 3.0, 0.5, 3.0, 0.1)),
    }
    assert counts["mean"] == counts["sum"]
    assert counts["mean_degree"] / counts["mean"] <= 1.02
    assert 0.8 <= counts["pna"] / counts["mean"] <= 1.2


def test_disjoint_union_preserves_segments() -> None:
    graph_a = generate_graph(n=64, feature_dimension=3, lambda_=8, rho=0.5, mu=0.2, seed=1, graph_id="a", condition="source")
    graph_b = generate_graph(n=64, feature_dimension=3, lambda_=8, rho=0.5, mu=0.2, seed=2, graph_id="b", condition="source")
    combined = combine_graphs([graph_a, graph_b], "combined")
    assert combined.n == 128
    assert torch.equal(torch.bincount(combined.edge_dst, minlength=128), combined.degree)
    assert bool(torch.all(combined.edge_dst[:-1] <= combined.edge_dst[1:]))
    assert not any(source < 64 <= target or target < 64 <= source for source, target in zip(combined.edge_src.tolist(), combined.edge_dst.tolist()))


def test_nested_bootstrap_reproducibility() -> None:
    values = np.arange(24, dtype=np.float64).reshape(4, 6) / 100.0
    first = nested_bootstrap(values, 100, 17)
    second = nested_bootstrap(values, 100, 17)
    assert np.array_equal(first, second)
    zero = nested_shift_bootstrap(values, values, 100, 29)
    assert abs(float(zero.mean())) < 0.05


def test_state_hash_sensitivity() -> None:
    model = make_model("mean")
    before = tensor_state_sha256(model.state_dict())
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    after = tensor_state_sha256(model.state_dict())
    assert before != after


def test_cuda_training_step_if_available() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    graph = generate_graph(
        n=4096,
        feature_dimension=8,
        lambda_=48,
        rho=0.8,
        mu=0.5,
        seed=stable_seed("cuda-kernel-preflight"),
        graph_id="cuda-kernel-preflight",
        condition="unit_test",
    )
    for name, width in (("mean", 64), ("mean_degree", 64), ("sum", 64), ("pna", 24)):
        torch.manual_seed(stable_seed("cuda-kernel-preflight", name))
        model = InductiveGNN(name, 8, width, 3.0, 0.5, 3.0, 0.0).to(device)
        logits = model(graph.x.to(device), graph.edge_src.to(device), graph.degree.to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, graph.y.to(device))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


TESTS = [
    test_config,
    test_environment_versions,
    test_seed_namespaces,
    test_triangle_sampler_all_edges_and_simple,
    test_edge_sampler_frequencies,
    test_environment_expected_degree,
    test_graph_reproducibility_balance_and_symmetry,
    test_aggregator_identities_and_permutation,
    test_model_interface_excludes_labels,
    test_parameter_budgets,
    test_disjoint_union_preserves_segments,
    test_nested_bootstrap_reproducibility,
    test_state_hash_sensitivity,
    test_cuda_training_step_if_available,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    records = []
    failed = False
    for test in TESTS:
        start = time.time()
        try:
            if test is test_config:
                test(config)
            else:
                test()
            records.append({"name": test.__name__, "status": "passed", "seconds": time.time() - start})
        except Exception as error:
            failed = True
            records.append({
                "name": test.__name__,
                "status": "failed",
                "seconds": time.time() - start,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            })
    report = {
        "status": "failed" if failed else "passed",
        "tests": records,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
