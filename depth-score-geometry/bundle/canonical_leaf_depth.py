from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Callable, Iterable

from canonical_depth import (
    CountSumLaw,
    Gradient,
    HiddenLaw,
    Mass,
    Matrix2,
    Number,
    add_mass,
    candidate_outcomes,
    convolve_candidate,
    generalized_eigendirections,
    generalized_eigenvalues,
    law_diagnostics,
    matrix_add,
    matrix_max_abs,
    matrix_relative_error,
    matrix_scale,
    matrix_subtract,
    outer,
    outer_over_variance,
    representation_fisher,
    round_ratio_ties_even,
    symmetric_eigenvalues,
)


def quantized_mean_update(count: int, index_sum: int, grid: int) -> int:
    if count == 0:
        return 0
    return min(grid, max(-grid, round_ratio_ties_even(index_sum, count)))


def leaf_base_law(theta0: Number, zero: Number, one: Number) -> HiddenLaw:
    two = one + one
    half = one / two
    return {
        -1: ((one - theta0) * half, zero, -half),
        1: ((one + theta0) * half, zero, half),
    }


def update_leaf_law(
    child: HiddenLaw,
    candidates: int,
    lambda0: Number,
    grid: int,
    zero: Number,
    one: Number,
) -> HiddenLaw:
    q = lambda0 / candidates
    dq: Gradient = (one / candidates, zero)
    if not (q > zero and q < one):
        raise ValueError("lambda/candidates must lie in (0,1)")
    joint: CountSumLaw = {(0, 0): (one, zero, zero)}
    outcomes = candidate_outcomes(child, q, dq, zero, one)
    for _ in range(candidates):
        joint = convolve_candidate(joint, outcomes, zero)
    output: dict[int, Mass] = {}
    for (count, index_sum), mass in joint.items():
        hidden_index = quantized_mean_update(count, index_sum, grid)
        output[hidden_index] = add_mass(
            output.get(hidden_index, (zero, zero, zero)), mass
        )
    return output


def leaf_laws_by_depth(
    depths: Iterable[int],
    candidates: int,
    lambda0: Number,
    theta0: Number,
    grid: int,
    zero: Number,
    one: Number,
) -> dict[int, HiddenLaw]:
    maximum = max(depths)
    current = {
        -grid: leaf_base_law(theta0, zero, one)[-1],
        grid: leaf_base_law(theta0, zero, one)[1],
    }
    result = {0: current}
    for depth in range(1, maximum + 1):
        current = update_leaf_law(
            current, candidates, lambda0, grid, zero, one
        )
        result[depth] = current
    return result


def probability_only_laws_by_depth(
    depths: Iterable[int],
    candidates: int,
    lambda0: float,
    theta0: float,
    grid: int,
) -> dict[int, dict[int, float]]:
    """Independent probability-only implementation used for finite differences."""
    current = {-grid: (1.0 - theta0) / 2.0, grid: (1.0 + theta0) / 2.0}
    result = {0: current}
    q = lambda0 / candidates
    if not 0.0 < q < 1.0:
        raise ValueError("lambda/candidates must lie in (0,1)")
    for depth in range(1, max(depths) + 1):
        joint: dict[tuple[int, int], float] = {(0, 0): 1.0}
        outcomes = [(0, 0, 1.0 - q), *[(1, h, q * p) for h, p in current.items()]]
        for _ in range(candidates):
            next_joint: dict[tuple[int, int], float] = {}
            for (count, index_sum), left in joint.items():
                for add_count, add_sum, right in outcomes:
                    key = count + add_count, index_sum + add_sum
                    next_joint[key] = next_joint.get(key, 0.0) + left * right
            joint = next_joint
        current = {}
        for (count, index_sum), probability in joint.items():
            hidden = quantized_mean_update(count, index_sum, grid)
            current[hidden] = current.get(hidden, 0.0) + probability
        result[depth] = current
    return result


def finite_difference_diagnostics(config: dict) -> tuple[float, float]:
    model = config["model"]
    channel = config["channels"]["quantized_recursive_mean"]
    steps = config["method"]["finite_difference_steps"]
    lambda0 = float(model["fiducial"]["lambda"])
    theta0 = float(model["fiducial"]["theta"])
    candidates = int(model["candidate_children"])
    depths = list(map(int, model["depths"]))
    maximum = 0.0
    maximum_fisher_relative_error = 0.0
    for delta in [channel["headline_delta"], *channel["resolution_controls"]]:
        grid = reciprocal_grid(float(delta))
        analytic = leaf_laws_by_depth(
            depths, candidates, lambda0, theta0, grid, 0.0, 1.0
        )
        perturbed = []
        for parameter, step in (("lambda", float(steps["lambda"])), ("theta", float(steps["theta"]))):
            plus_lambda = lambda0 + step if parameter == "lambda" else lambda0
            minus_lambda = lambda0 - step if parameter == "lambda" else lambda0
            plus_theta = theta0 + step if parameter == "theta" else theta0
            minus_theta = theta0 - step if parameter == "theta" else theta0
            plus = probability_only_laws_by_depth(
                depths, candidates, plus_lambda, plus_theta, grid
            )
            minus = probability_only_laws_by_depth(
                depths, candidates, minus_lambda, minus_theta, grid
            )
            perturbed.append((step, plus, minus))
        for depth in depths:
            support = set(analytic[depth])
            for _, plus, minus in perturbed:
                support.update(plus[depth])
                support.update(minus[depth])
            finite_difference_scores: dict[int, list[float]] = {
                hidden: [] for hidden in support
            }
            for hidden in support:
                mass = analytic[depth].get(hidden, (0.0, 0.0, 0.0))
                for derivative_index, (step, plus, minus) in enumerate(perturbed, start=1):
                    finite_difference = (
                        plus[depth].get(hidden, 0.0)
                        - minus[depth].get(hidden, 0.0)
                    ) / (2.0 * step)
                    maximum = max(
                        maximum, abs(float(mass[derivative_index]) - finite_difference)
                    )
                    finite_difference_scores[hidden].append(finite_difference)
            finite_difference_fisher: Matrix2 = ((0.0, 0.0), (0.0, 0.0))
            for hidden, derivative in finite_difference_scores.items():
                probability = float(analytic[depth][hidden][0])
                if probability <= 0.0:
                    raise ValueError("nonpositive nominal state in finite difference")
                contribution: Matrix2 = (
                    (
                        derivative[0] * derivative[0] / probability,
                        derivative[0] * derivative[1] / probability,
                    ),
                    (
                        derivative[0] * derivative[1] / probability,
                        derivative[1] * derivative[1] / probability,
                    ),
                )
                finite_difference_fisher = matrix_add(
                    finite_difference_fisher, contribution
                )
            analytic_fisher = representation_fisher(analytic[depth], 0.0)
            maximum_fisher_relative_error = max(
                maximum_fisher_relative_error,
                matrix_relative_error(finite_difference_fisher, analytic_fisher),
            )
    return maximum, maximum_fisher_relative_error


def raw_leaf_fisher_by_depth(
    depths: Iterable[int],
    candidates: int,
    lambda0: Number,
    theta0: Number,
    zero: Number,
    one: Number,
) -> dict[int, Matrix2]:
    maximum = max(depths)
    current: Matrix2 = (
        (zero, zero),
        (zero, one / (one - theta0 * theta0)),
    )
    result = {0: current}
    q = lambda0 / candidates
    dq: Gradient = (one / candidates, zero)
    local = matrix_scale(
        one * candidates, outer_over_variance(dq, q, one)
    )
    for depth in range(1, maximum + 1):
        current = matrix_add(local, matrix_scale(lambda0, current))
        result[depth] = current
    return result


def score_leaf_fisher_by_depth(
    depths: Iterable[int],
    candidates: int,
    lambda0: Number,
    theta0: Number,
    zero: Number,
    one: Number,
) -> dict[int, Matrix2]:
    maximum = max(depths)
    base_law = leaf_base_law(theta0, zero, one)
    current: Matrix2 = ((zero, zero), (zero, zero))
    for hidden, (probability, _, _) in base_law.items():
        mark_score: Gradient = (zero, hidden * one / (one + hidden * theta0))
        current = matrix_add(current, matrix_scale(probability, outer(mark_score)))
    result = {0: current}
    q = lambda0 / candidates
    dq: Gradient = (one / candidates, zero)
    for depth in range(1, maximum + 1):
        absent_score: Gradient = (-dq[0] / (one - q), zero)
        present_score: Gradient = (dq[0] / q, zero)
        candidate_second = matrix_add(
            matrix_scale(one - q, outer(absent_score)),
            matrix_scale(q, matrix_add(outer(present_score), current)),
        )
        current = matrix_scale(one * candidates, candidate_second)
        result[depth] = current
    return result


def reciprocal_grid(delta: float) -> int:
    reciprocal = 1.0 / delta
    grid = round(reciprocal)
    if abs(reciprocal - grid) > 1e-12:
        raise ValueError(f"delta is not an integer reciprocal: {delta}")
    return grid


def calculate(config: dict, factory: Callable[[str], Number]) -> dict:
    model = config["model"]
    channel = config["channels"]["quantized_recursive_mean"]
    zero, one = factory("0"), factory("1")
    lambda0 = factory(str(model["fiducial"]["lambda"]))
    theta0 = factory(str(model["fiducial"]["theta"]))
    candidates = int(model["candidate_children"])
    depths = list(map(int, model["depths"]))
    deltas = [channel["headline_delta"], *channel["resolution_controls"]]
    raw = raw_leaf_fisher_by_depth(
        depths, candidates, lambda0, theta0, zero, one
    )
    score = score_leaf_fisher_by_depth(
        depths, candidates, lambda0, theta0, zero, one
    )
    records = []
    matrices = []
    for raw_delta in deltas:
        grid = reciprocal_grid(float(raw_delta))
        laws = leaf_laws_by_depth(
            depths, candidates, lambda0, theta0, grid, zero, one
        )
        for depth in depths:
            law = laws[depth]
            representation = representation_fisher(law, zero)
            raw_matrix = raw[depth]
            score_matrix = score[depth]
            eigenvalues = generalized_eigenvalues(representation, raw_matrix)
            eigendirections = generalized_eigendirections(
                representation, raw_matrix, eigenvalues
            )
            score_eigenvalues = generalized_eigenvalues(score_matrix, raw_matrix)
            representation_psd = symmetric_eigenvalues(representation)
            contraction_psd = symmetric_eigenvalues(
                matrix_subtract(raw_matrix, representation)
            )
            diagnostics = law_diagnostics(law)
            efficiencies = [
                float(representation[0][0] / raw_matrix[0][0]),
                float(representation[1][1] / raw_matrix[1][1]),
            ]
            for index, direction in enumerate(("intensity", "leaf_location")):
                records.append(
                    {
                        "delta": float(raw_delta),
                        "depth": depth,
                        "direction": direction,
                        "raw_information": float(raw_matrix[index][index]),
                        "representation_information": float(
                            representation[index][index]
                        ),
                        "efficiency": efficiencies[index],
                        **diagnostics,
                    }
                )
            matrices.append(
                {
                    "delta": float(raw_delta),
                    "depth": depth,
                    "grid": grid,
                    "raw_fisher": [[float(v) for v in row] for row in raw_matrix],
                    "representation_fisher": [
                        [float(v) for v in row] for row in representation
                    ],
                    "oracle_score_fisher": [
                        [float(v) for v in row] for row in score_matrix
                    ],
                    "generalized_eigenvalues": list(eigenvalues),
                    "generalized_eigendirections": eigendirections,
                    "oracle_score_generalized_eigenvalues": list(score_eigenvalues),
                    "representation_psd_eigenvalues": list(representation_psd),
                    "contraction_psd_eigenvalues": list(contraction_psd),
                    "directional_efficiencies": efficiencies,
                    "raw_fisher_recursion_relative_error": matrix_relative_error(
                        score_matrix, raw_matrix
                    ),
                    "diagnostics": diagnostics,
                }
            )
    return {"records": records, "matrices": matrices}


def compare_precision(float_result: dict, decimal_result: dict) -> float:
    if len(float_result["matrices"]) != len(decimal_result["matrices"]):
        raise ValueError("float and Decimal runs returned different matrix counts")
    maximum = 0.0
    for left, right in zip(float_result["matrices"], decimal_result["matrices"]):
        for key in ("delta", "depth", "grid"):
            if left[key] != right[key]:
                raise ValueError(f"float/Decimal metadata mismatch for {key}")
        for key in ("raw_fisher", "representation_fisher", "oracle_score_fisher"):
            left_matrix = left[key]
            right_matrix = right[key]
            if (
                len(left_matrix) != 2
                or len(right_matrix) != 2
                or any(len(row) != 2 for row in left_matrix)
                or any(len(row) != 2 for row in right_matrix)
            ):
                raise ValueError(f"float/Decimal matrix shape mismatch for {key}")
            for row_index in range(2):
                for column_index in range(2):
                    maximum = max(
                        maximum,
                        abs(
                            left_matrix[row_index][column_index]
                            - right_matrix[row_index][column_index]
                        ),
                    )
    return maximum


def eigendirection_certificates(matrix: dict) -> tuple[float, float]:
    raw = matrix["raw_fisher"]
    representation = matrix["representation_fisher"]
    max_residual = 0.0
    max_normalization_error = 0.0
    for eigenvalue, item in zip(
        matrix["generalized_eigenvalues"], matrix["generalized_eigendirections"]
    ):
        vector = item["raw_fisher_normalized_direction"]
        residual = [
            representation[0][0] * vector[0]
            + representation[0][1] * vector[1]
            - eigenvalue * (raw[0][0] * vector[0] + raw[0][1] * vector[1]),
            representation[1][0] * vector[0]
            + representation[1][1] * vector[1]
            - eigenvalue * (raw[1][0] * vector[0] + raw[1][1] * vector[1]),
        ]
        max_residual = max(max_residual, *(abs(value) for value in residual))
        norm = (
            raw[0][0] * vector[0] * vector[0]
            + 2.0 * raw[0][1] * vector[0] * vector[1]
            + raw[1][1] * vector[1] * vector[1]
        )
        max_normalization_error = max(max_normalization_error, abs(norm - 1.0))
    return max_residual, max_normalization_error


def evaluate_gates(
    config: dict,
    result: dict,
    precision_error: float,
    finite_difference_error: float,
    finite_difference_fisher_error: float,
    test_suite_passed: bool,
) -> dict:
    gates = config["validity_gates"]
    headline = float(config["channels"]["quantized_recursive_mean"]["headline_delta"])
    matrices = result["matrices"]
    expected_depths = {1, 2, 3, 4}
    configured_depths = set(map(int, config["model"]["depths"]))
    expected_deltas = {
        headline,
        *map(
            float,
            config["channels"]["quantized_recursive_mean"]["resolution_controls"],
        ),
    }
    observed_pairs = [(row["delta"], row["depth"]) for row in matrices]
    expected_pairs = {
        (delta, depth) for delta in expected_deltas for depth in expected_depths
    }
    complete_design = (
        configured_depths == expected_depths
        and len(observed_pairs) == len(expected_pairs)
        and set(observed_pairs) == expected_pairs
        and len(set(observed_pairs)) == len(observed_pairs)
    )
    numeric_values = []
    for row in matrices:
        numeric_values.extend(row["generalized_eigenvalues"])
        numeric_values.extend(row["directional_efficiencies"])
        numeric_values.extend(value for matrix_row in row["raw_fisher"] for value in matrix_row)
        numeric_values.extend(
            value for matrix_row in row["representation_fisher"] for value in matrix_row
        )
        numeric_values.extend(
            value for matrix_row in row["oracle_score_fisher"] for value in matrix_row
        )
    finite_outputs = all(math.isfinite(float(value)) for value in numeric_values)
    certificates = [eigendirection_certificates(matrix) for matrix in matrices]
    statistics = {
        "probability_mass_error_max": max(
            row["diagnostics"]["probability_mass_error"] for row in matrices
        ),
        "score_mean_max_abs": max(
            row["diagnostics"]["score_mean_max_abs"] for row in matrices
        ),
        "raw_fisher_recursion_relative_error_max": max(
            row["raw_fisher_recursion_relative_error"] for row in matrices
        ),
        "finite_difference_derivative_max_abs_error": finite_difference_error,
        "finite_difference_fisher_relative_error_max": finite_difference_fisher_error,
        "eigendirection_residual_max": max(value[0] for value in certificates),
        "eigendirection_normalization_max_abs_error": max(
            value[1] for value in certificates
        ),
        "representation_psd_min_eigenvalue": min(
            min(row["representation_psd_eigenvalues"]) for row in matrices
        ),
        "contraction_psd_min_eigenvalue": min(
            min(row["contraction_psd_eigenvalues"]) for row in matrices
        ),
        "data_processing_max_eigenvalue": max(
            max(row["generalized_eigenvalues"]) for row in matrices
        ),
        "high_precision_matrix_max_abs_discrepancy": precision_error,
        "expected_result_cells": len(expected_pairs),
        "observed_result_cells": len(observed_pairs),
    }
    validity = {
        "test_suite": test_suite_passed,
        "complete_design": complete_design,
        "finite_outputs": finite_outputs,
        "probability_mass": statistics["probability_mass_error_max"]
        <= gates["probability_mass_error_max"],
        "score_mean": statistics["score_mean_max_abs"]
        <= gates["score_mean_max_abs"],
        "raw_fisher_recursion": statistics[
            "raw_fisher_recursion_relative_error_max"
        ]
        <= gates["raw_fisher_recursion_relative_error_max"],
        "finite_difference_derivative": statistics[
            "finite_difference_derivative_max_abs_error"
        ]
        <= gates["finite_difference_derivative_max_abs_error"],
        "finite_difference_fisher": statistics[
            "finite_difference_fisher_relative_error_max"
        ]
        <= gates["finite_difference_fisher_relative_error_max"],
        "eigendirection_residual": statistics["eigendirection_residual_max"]
        <= gates["eigendirection_residual_max"],
        "eigendirection_normalization": statistics[
            "eigendirection_normalization_max_abs_error"
        ]
        <= gates["eigendirection_normalization_max_abs_error"],
        "representation_psd": statistics["representation_psd_min_eigenvalue"]
        >= gates["fisher_psd_min_eigenvalue"],
        "contraction_psd": statistics["contraction_psd_min_eigenvalue"]
        >= gates["fisher_psd_min_eigenvalue"],
        "data_processing": statistics["data_processing_max_eigenvalue"]
        <= gates["data_processing_max_eigenvalue"],
        "high_precision": statistics["high_precision_matrix_max_abs_discrepancy"]
        <= gates["high_precision_matrix_max_abs_discrepancy"],
    }
    headline_rows = [
        row for row in matrices if row["delta"] == headline and row["depth"] >= 2
    ]
    headline_complete = {row["depth"] for row in headline_rows} == {2, 3, 4}
    if not headline_complete or len(headline_rows) != 3:
        headline_rows = []
    headline_gaps = {
        str(row["depth"]): row["directional_efficiencies"][1]
        - row["directional_efficiencies"][0]
        for row in headline_rows
    }
    all_gaps: dict[int, list[float]] = defaultdict(list)
    for row in matrices:
        if row["depth"] >= 2:
            all_gaps[row["depth"]].append(
                row["directional_efficiencies"][1]
                - row["directional_efficiencies"][0]
            )
    h1_error = max(
        abs(value - 1.0)
        for row in matrices
        for value in row["oracle_score_generalized_eigenvalues"]
    )
    hypotheses = {
        "H1": h1_error <= gates["raw_fisher_recursion_relative_error_max"],
        "H2": headline_complete
        and len(headline_gaps) == 3
        and all(gap > 0.10 for gap in headline_gaps.values()),
        "H3": set(all_gaps) == {2, 3, 4}
        and all(len(gaps) == 3 for gaps in all_gaps.values())
        and all(all(gap > 0.0 for gap in gaps) for gaps in all_gaps.values()),
    }
    return {
        "validity": validity,
        "validity_statistics": statistics,
        "hypotheses": hypotheses,
        "hypothesis_statistics": {
            "H1_oracle_max_abs_efficiency_error": h1_error,
            "H2_headline_leaf_location_minus_intensity_by_depth": headline_gaps,
            "H3_all_resolution_leaf_location_minus_intensity_by_depth": {
                str(depth): gaps for depth, gaps in sorted(all_gaps.items())
            },
        },
        "scientific_success": all(validity.values()) and all(hypotheses.values()),
    }


def write_outputs(
    output: Path,
    config: dict,
    result: dict,
    precision_error: float,
    finite_difference_error: float,
    finite_difference_fisher_error: float,
    test_report: dict,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with (output / "canonical_leaf_depth.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["records"][0]))
        writer.writeheader()
        writer.writerows(result["records"])
    report = {
        "experiment_id": config["experiment_id"],
        "precision_matrix_max_abs_discrepancy": precision_error,
        "mechanical_test_report": test_report,
        "matrices": result["matrices"],
    }
    report.update(
        evaluate_gates(
            config,
            result,
            precision_error,
            finite_difference_error,
            finite_difference_fisher_error,
            bool(test_report["passed"]),
        )
    )
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decimal-precision", type=int, default=80)
    args = parser.parse_args()
    test_path = Path(__file__).with_name("test_canonical_leaf_depth.py")
    test_environment = dict(os.environ)
    test_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed_test = subprocess.run(
        [sys.executable, str(test_path)],
        cwd=str(test_path.parent),
        env=test_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    test_report = {
        "passed": completed_test.returncode == 0,
        "returncode": completed_test.returncode,
        "stdout": completed_test.stdout.strip(),
        "stderr": completed_test.stderr.strip(),
    }
    if not test_report["passed"]:
        raise RuntimeError(f"mechanical test suite failed: {test_report}")
    config = json.loads(args.config.read_text())
    float_result = calculate(config, float)
    with localcontext() as context:
        context.prec = args.decimal_precision
        decimal_result = calculate(config, Decimal)
    precision_error = compare_precision(float_result, decimal_result)
    derivative_error, derivative_fisher_error = finite_difference_diagnostics(config)
    write_outputs(
        args.output,
        config,
        float_result,
        precision_error,
        derivative_error,
        derivative_fisher_error,
        test_report,
    )


if __name__ == "__main__":
    main()
