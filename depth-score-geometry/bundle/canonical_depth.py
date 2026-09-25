from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Callable, Iterable, TypeVar


Number = TypeVar("Number", float, Decimal)
Gradient = tuple[Number, Number]
Mass = tuple[Number, Number, Number]
HiddenLaw = dict[int, Mass]
CountSumLaw = dict[tuple[int, int], Mass]
Matrix2 = tuple[tuple[Number, Number], tuple[Number, Number]]


def add_mass(left: Mass, right: Mass) -> Mass:
    return tuple(a + b for a, b in zip(left, right))  # type: ignore[return-value]


def mul_mass(left: Mass, right: Mass) -> Mass:
    p, dl, dr = left
    q, ql, qr = right
    return p * q, dl * q + p * ql, dr * q + p * qr


def round_ratio_ties_even(numerator: int, denominator: int) -> int:
    """Round an exact rational to the nearest integer with ties to even."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    lower, remainder = divmod(numerator, denominator)
    doubled = 2 * remainder
    if doubled < denominator:
        return lower
    if doubled > denominator:
        return lower + 1
    return lower if lower % 2 == 0 else lower + 1


def quantized_update(parent_type: int, count: int, index_sum: int, grid: int) -> int:
    """Q((a + mean child hidden)/2) in integer grid coordinates."""
    if parent_type not in (-1, 1):
        raise ValueError("parent type must be -1 or 1")
    if count == 0:
        numerator, denominator = grid * parent_type, 2
    else:
        numerator = grid * parent_type * count + index_sum
        denominator = 2 * count
    return min(grid, max(-grid, round_ratio_ties_even(numerator, denominator)))


def edge_quantities(
    parent_type: int,
    child_type: int,
    lambda0: Number,
    rho0: Number,
    total_candidates: int,
    one: Number,
) -> tuple[Number, Gradient]:
    sign = parent_type * child_type
    q = lambda0 * (one + rho0 * sign) / total_candidates
    dq_lambda = (one + rho0 * sign) / total_candidates
    dq_rho = lambda0 * sign / total_candidates
    if not (q > 0 and q < one):
        raise ValueError(f"edge probability outside (0,1): {q}")
    return q, (dq_lambda, dq_rho)


def candidate_outcomes(
    child: HiddenLaw,
    q: Number,
    dq: Gradient,
    zero: Number,
    one: Number,
) -> list[tuple[int, int, Mass]]:
    outcomes: list[tuple[int, int, Mass]] = [
        (0, 0, (one - q, -dq[0], -dq[1]))
    ]
    for hidden_index, (p, dl, dr) in child.items():
        outcomes.append(
            (
                1,
                hidden_index,
                (q * p, dq[0] * p + q * dl, dq[1] * p + q * dr),
            )
        )
    return outcomes


def convolve_candidate(
    law: CountSumLaw,
    outcomes: Iterable[tuple[int, int, Mass]],
    zero: Number,
) -> CountSumLaw:
    result: dict[tuple[int, int], Mass] = {}
    outcomes = list(outcomes)
    for (count, index_sum), left in law.items():
        for add_count, add_sum, right in outcomes:
            key = count + add_count, index_sum + add_sum
            contribution = mul_mass(left, right)
            previous = result.get(key, (zero, zero, zero))
            result[key] = add_mass(previous, contribution)
    return result


def update_type_law(
    parent_type: int,
    child_laws: dict[int, HiddenLaw],
    candidates_per_type: int,
    lambda0: Number,
    rho0: Number,
    grid: int,
    zero: Number,
    one: Number,
) -> HiddenLaw:
    total_candidates = 2 * candidates_per_type
    joint: CountSumLaw = {(0, 0): (one, zero, zero)}
    for child_type in (-1, 1):
        q, dq = edge_quantities(
            parent_type, child_type, lambda0, rho0, total_candidates, one
        )
        outcomes = candidate_outcomes(child_laws[child_type], q, dq, zero, one)
        for _ in range(candidates_per_type):
            joint = convolve_candidate(joint, outcomes, zero)

    output: dict[int, Mass] = {}
    for (count, index_sum), value in joint.items():
        hidden_index = quantized_update(parent_type, count, index_sum, grid)
        previous = output.get(hidden_index, (zero, zero, zero))
        output[hidden_index] = add_mass(previous, value)
    return output


def hidden_laws_by_depth(
    depths: Iterable[int],
    candidates_per_type: int,
    lambda0: Number,
    rho0: Number,
    grid: int,
    zero: Number,
    one: Number,
) -> dict[int, dict[int, HiddenLaw]]:
    maximum = max(depths)
    current: dict[int, HiddenLaw] = {
        -1: {-grid: (one, zero, zero)},
        1: {grid: (one, zero, zero)},
    }
    result = {0: current}
    for depth in range(1, maximum + 1):
        current = {
            parent_type: update_type_law(
                parent_type,
                current,
                candidates_per_type,
                lambda0,
                rho0,
                grid,
                zero,
                one,
            )
            for parent_type in (-1, 1)
        }
        result[depth] = current
    return result


def representation_fisher(law: HiddenLaw, zero: Number) -> Matrix2:
    f00 = f01 = f11 = zero
    for p, dl, dr in law.values():
        if p <= zero:
            raise ValueError("nonpositive output-state probability")
        f00 += dl * dl / p
        f01 += dl * dr / p
        f11 += dr * dr / p
    return ((f00, f01), (f01, f11))


def matrix_add(left: Matrix2, right: Matrix2) -> Matrix2:
    return (
        (left[0][0] + right[0][0], left[0][1] + right[0][1]),
        (left[1][0] + right[1][0], left[1][1] + right[1][1]),
    )


def matrix_scale(value: Number, matrix: Matrix2) -> Matrix2:
    return (
        (value * matrix[0][0], value * matrix[0][1]),
        (value * matrix[1][0], value * matrix[1][1]),
    )


def outer_over_variance(dq: Gradient, q: Number, one: Number) -> Matrix2:
    scale = one / (q * (one - q))
    return (
        (scale * dq[0] * dq[0], scale * dq[0] * dq[1]),
        (scale * dq[0] * dq[1], scale * dq[1] * dq[1]),
    )


def raw_fisher_by_depth(
    depths: Iterable[int],
    candidates_per_type: int,
    lambda0: Number,
    rho0: Number,
    zero: Number,
    one: Number,
) -> dict[int, dict[int, Matrix2]]:
    maximum = max(depths)
    zero_matrix: Matrix2 = ((zero, zero), (zero, zero))
    current = {-1: zero_matrix, 1: zero_matrix}
    result = {0: current}
    total_candidates = 2 * candidates_per_type
    for depth in range(1, maximum + 1):
        next_value: dict[int, Matrix2] = {}
        for parent_type in (-1, 1):
            local = zero_matrix
            descendants = zero_matrix
            for child_type in (-1, 1):
                q, dq = edge_quantities(
                    parent_type, child_type, lambda0, rho0, total_candidates, one
                )
                local = matrix_add(
                    local,
                    matrix_scale(
                        one * candidates_per_type,
                        outer_over_variance(dq, q, one),
                    ),
                )
                descendants = matrix_add(
                    descendants,
                    matrix_scale(one * candidates_per_type * q, current[child_type]),
                )
            next_value[parent_type] = matrix_add(local, descendants)
        current = next_value
        result[depth] = current
    return result


def outer(vector: Gradient) -> Matrix2:
    return (
        (vector[0] * vector[0], vector[0] * vector[1]),
        (vector[0] * vector[1], vector[1] * vector[1]),
    )


def score_moment_fisher_by_depth(
    depths: Iterable[int],
    candidates_per_type: int,
    lambda0: Number,
    rho0: Number,
    zero: Number,
    one: Number,
) -> dict[int, dict[int, Matrix2]]:
    """Second moment of the complete score via candidate-outcome mixing.

    This deliberately implements the score channel independently of
    ``raw_fisher_by_depth``.  For one candidate, the absent score is
    ``-dq/(1-q)`` and the present score is ``dq/q + S_child``.  Candidate
    scores have mean zero, so their second moments add across the finite tree.
    The result is the canonical Fisher matrix of both oracle score channels,
    because their output is the complete score itself.
    """
    maximum = max(depths)
    zero_matrix: Matrix2 = ((zero, zero), (zero, zero))
    current = {-1: zero_matrix, 1: zero_matrix}
    result = {0: current}
    total_candidates = 2 * candidates_per_type
    for depth in range(1, maximum + 1):
        next_value: dict[int, Matrix2] = {}
        for parent_type in (-1, 1):
            parent_second_moment = zero_matrix
            for child_type in (-1, 1):
                q, dq = edge_quantities(
                    parent_type, child_type, lambda0, rho0, total_candidates, one
                )
                absent_score = (-dq[0] / (one - q), -dq[1] / (one - q))
                present_score = (dq[0] / q, dq[1] / q)
                candidate_second_moment = matrix_add(
                    matrix_scale(one - q, outer(absent_score)),
                    matrix_scale(
                        q,
                        matrix_add(outer(present_score), current[child_type]),
                    ),
                )
                parent_second_moment = matrix_add(
                    parent_second_moment,
                    matrix_scale(one * candidates_per_type, candidate_second_moment),
                )
            next_value[parent_type] = parent_second_moment
        current = next_value
        result[depth] = current
    return result


def directional_efficiency(representation: Matrix2, raw: Matrix2, index: int) -> float:
    return float(representation[index][index] / raw[index][index])


def generalized_eigenvalues(representation: Matrix2, raw: Matrix2) -> tuple[float, float]:
    a, b, c = map(float, (raw[0][0], raw[0][1], raw[1][1]))
    x, y, z = map(float, (representation[0][0], representation[0][1], representation[1][1]))
    determinant_raw = a * c - b * b
    determinant_representation = x * z - y * y
    if determinant_raw <= 0:
        raise ValueError("raw Fisher matrix is not positive definite")
    linear = x * c + z * a - 2.0 * y * b
    discriminant = max(0.0, linear * linear - 4.0 * determinant_raw * determinant_representation)
    low = (linear - math.sqrt(discriminant)) / (2.0 * determinant_raw)
    high = (linear + math.sqrt(discriminant)) / (2.0 * determinant_raw)
    return low, high


def generalized_eigendirections(
    representation: Matrix2, raw: Matrix2, eigenvalues: tuple[float, float]
) -> list[dict[str, list[float]]]:
    """Return raw-Fisher-normalized directions and coordinate alignments."""
    raw_float = tuple(tuple(float(value) for value in row) for row in raw)
    representation_float = tuple(
        tuple(float(value) for value in row) for row in representation
    )
    output = []
    for eigenvalue in eigenvalues:
        m00 = representation_float[0][0] - eigenvalue * raw_float[0][0]
        m01 = representation_float[0][1] - eigenvalue * raw_float[0][1]
        m11 = representation_float[1][1] - eigenvalue * raw_float[1][1]
        if m00 * m00 + m01 * m01 >= m01 * m01 + m11 * m11:
            vector = [-m01, m00]
        else:
            vector = [-m11, m01]
        if abs(vector[0]) + abs(vector[1]) < 1e-14:
            vector = [1.0, 0.0]
        norm_squared = (
            raw_float[0][0] * vector[0] * vector[0]
            + 2.0 * raw_float[0][1] * vector[0] * vector[1]
            + raw_float[1][1] * vector[1] * vector[1]
        )
        scale = math.sqrt(max(norm_squared, 1e-300))
        vector = [value / scale for value in vector]
        if vector[0] < 0.0 or (vector[0] == 0.0 and vector[1] < 0.0):
            vector = [-value for value in vector]
        fisher_products = [
            raw_float[0][0] * vector[0] + raw_float[0][1] * vector[1],
            raw_float[0][1] * vector[0] + raw_float[1][1] * vector[1],
        ]
        alignments = [
            abs(fisher_products[0]) / math.sqrt(raw_float[0][0]),
            abs(fisher_products[1]) / math.sqrt(raw_float[1][1]),
        ]
        output.append(
            {
                "raw_fisher_normalized_direction": vector,
                "coordinate_fisher_cosines": alignments,
            }
        )
    return output


def symmetric_eigenvalues(matrix: Matrix2) -> tuple[float, float]:
    a, b, c = map(float, (matrix[0][0], matrix[0][1], matrix[1][1]))
    radius = math.sqrt(max(0.0, (a - c) * (a - c) + 4.0 * b * b))
    return (0.5 * (a + c - radius), 0.5 * (a + c + radius))


def matrix_subtract(left: Matrix2, right: Matrix2) -> Matrix2:
    return (
        (left[0][0] - right[0][0], left[0][1] - right[0][1]),
        (left[1][0] - right[1][0], left[1][1] - right[1][1]),
    )


def matrix_max_abs(matrix: Matrix2) -> float:
    return max(abs(float(value)) for row in matrix for value in row)


def matrix_relative_error(left: Matrix2, right: Matrix2) -> float:
    denominator = max(matrix_max_abs(right), 1e-300)
    return matrix_max_abs(matrix_subtract(left, right)) / denominator


def law_diagnostics(law: HiddenLaw) -> dict[str, float]:
    total = sum(float(value[0]) for value in law.values())
    score_lambda = sum(float(value[1]) for value in law.values())
    score_rho = sum(float(value[2]) for value in law.values())
    return {
        "probability_mass_error": abs(total - 1.0),
        "score_mean_max_abs": max(abs(score_lambda), abs(score_rho)),
        "support_size": len(law),
    }


def calculate(config: dict, factory: Callable[[str], Number]) -> dict:
    model = config["model"]
    channel = config["channels"]["quantized_recursive_mean"]
    lambda0 = factory(str(model["fiducial"]["lambda"]))
    rho0 = factory(str(model["fiducial"]["rho"]))
    zero, one = factory("0"), factory("1")
    depths = list(map(int, model["depths"]))
    candidates = int(model["candidate_children_per_type"])
    deltas = [channel["headline_delta"], *channel["resolution_controls"]]
    raw = raw_fisher_by_depth(depths, candidates, lambda0, rho0, zero, one)
    score_moment = score_moment_fisher_by_depth(
        depths, candidates, lambda0, rho0, zero, one
    )
    records = []
    matrices = []
    for raw_delta in deltas:
        grid = round(1.0 / float(raw_delta))
        laws = hidden_laws_by_depth(
            depths, candidates, lambda0, rho0, grid, zero, one
        )
        for depth in depths:
            law = laws[depth][int(model["root_type"])]
            representation = representation_fisher(law, zero)
            raw_matrix = raw[depth][int(model["root_type"])]
            eigenvalues = generalized_eigenvalues(representation, raw_matrix)
            eigendirections = generalized_eigendirections(
                representation, raw_matrix, eigenvalues
            )
            representation_psd = symmetric_eigenvalues(representation)
            contraction_psd = symmetric_eigenvalues(
                matrix_subtract(raw_matrix, representation)
            )
            score_matrix = score_moment[depth][int(model["root_type"])]
            raw_recursion_error = matrix_relative_error(score_matrix, raw_matrix)
            score_eigenvalues = generalized_eigenvalues(score_matrix, raw_matrix)
            diagnostics = law_diagnostics(law)
            efficiencies = [
                directional_efficiency(representation, raw_matrix, 0),
                directional_efficiency(representation, raw_matrix, 1),
            ]
            for index, direction in enumerate(("intensity", "mixing")):
                records.append(
                    {
                        "delta": float(raw_delta),
                        "depth": depth,
                        "direction": direction,
                        "raw_information": float(raw_matrix[index][index]),
                        "representation_information": float(representation[index][index]),
                        "efficiency": efficiencies[index],
                        **diagnostics,
                    }
                )
            matrices.append(
                {
                    "delta": float(raw_delta),
                    "depth": depth,
                    "raw_fisher": [[float(item) for item in row] for row in raw_matrix],
                    "representation_fisher": [
                        [float(item) for item in row] for row in representation
                    ],
                    "generalized_eigenvalues": list(eigenvalues),
                    "generalized_eigendirections": eigendirections,
                    "representation_psd_eigenvalues": list(representation_psd),
                    "contraction_psd_eigenvalues": list(contraction_psd),
                    "directional_efficiencies": efficiencies,
                    "oracle_score_sum_fisher": [
                        [float(item) for item in row] for row in score_matrix
                    ],
                    "oracle_degree_mean_fisher": [
                        [float(item) for item in row] for row in score_matrix
                    ],
                    "oracle_score_generalized_eigenvalues": list(score_eigenvalues),
                    "raw_fisher_recursion_relative_error": raw_recursion_error,
                    "diagnostics": diagnostics,
                }
            )
    return {"records": records, "matrices": matrices}


def compare_precision(float_result: dict, decimal_result: dict) -> float:
    left = float_result["matrices"]
    right = decimal_result["matrices"]
    if len(left) != len(right):
        raise ValueError("precision runs returned different matrix counts")
    maximum = 0.0
    for a, b in zip(left, right):
        for key in (
            "raw_fisher",
            "representation_fisher",
            "oracle_score_sum_fisher",
            "oracle_degree_mean_fisher",
        ):
            for row_a, row_b in zip(a[key], b[key]):
                maximum = max(maximum, *(abs(x - y) for x, y in zip(row_a, row_b)))
    return maximum


def evaluate_gates(config: dict, result: dict, precision_error: float) -> dict:
    gates = config["validity_gates"]
    headline = float(config["channels"]["quantized_recursive_mean"]["headline_delta"])
    matrices = result["matrices"]
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
    }
    validity = {
        "probability_mass": statistics["probability_mass_error_max"]
        <= gates["probability_mass_error_max"],
        "score_mean": statistics["score_mean_max_abs"]
        <= gates["score_mean_max_abs"],
        "raw_fisher_recursion": statistics[
            "raw_fisher_recursion_relative_error_max"
        ]
        <= gates["raw_fisher_recursion_relative_error_max"],
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
    headline_gaps = {
        str(row["depth"]): row["directional_efficiencies"][1]
        - row["directional_efficiencies"][0]
        for row in headline_rows
    }
    h2 = all(
        gap > 0.10 for gap in headline_gaps.values()
    )
    by_depth: dict[int, list[float]] = defaultdict(list)
    for row in matrices:
        if row["depth"] >= 2:
            by_depth[row["depth"]].append(
                row["directional_efficiencies"][1] - row["directional_efficiencies"][0]
            )
    h3 = all(all(gap > 0.0 for gap in gaps) for gaps in by_depth.values())
    h1 = all(
        max(abs(value - 1.0) for value in row["oracle_score_generalized_eigenvalues"])
        <= gates["raw_fisher_recursion_relative_error_max"]
        for row in matrices
    )
    return {
        "validity": validity,
        "validity_statistics": statistics,
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3},
        "hypothesis_statistics": {
            "H1_oracle_max_abs_efficiency_error": max(
                abs(value - 1.0)
                for row in matrices
                for value in row["oracle_score_generalized_eigenvalues"]
            ),
            "H2_headline_mixing_minus_intensity_by_depth": headline_gaps,
            "H3_all_resolution_mixing_minus_intensity_by_depth": {
                str(depth): gaps for depth, gaps in sorted(by_depth.items())
            },
        },
        "scientific_success": all(validity.values()) and h1 and h2 and h3,
    }


def write_outputs(output: Path, config: dict, result: dict, precision_error: float) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with (output / "canonical_depth.csv").open("w", newline="") as handle:
        fieldnames = list(result["records"][0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["records"])
    report = {
        "experiment_id": config["experiment_id"],
        "precision_matrix_max_abs_discrepancy": precision_error,
        "matrices": result["matrices"],
    }
    report.update(evaluate_gates(config, result, precision_error))
    (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decimal-precision", type=int, default=80)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    float_result = calculate(config, float)
    with localcontext() as context:
        context.prec = args.decimal_precision
        decimal_result = calculate(config, Decimal)
    error = compare_precision(float_result, decimal_result)
    write_outputs(args.output, config, float_result, error)


if __name__ == "__main__":
    main()
