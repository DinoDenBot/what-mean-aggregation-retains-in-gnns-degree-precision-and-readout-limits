#!/usr/bin/env python3
"""Exact rate-Fisher lower bound from the implemented float32 label mean.

The finite-CSBM generator draws independent same- and cross-class counts.  Its
mean aggregate contains the float32 value (N_same-N_cross)/(N_same+N_cross),
with a distinct empty-neighborhood symbol.  Grouping the finite count support
by that value gives the exact conditional rate score and hence a lower bound
on the Fisher information of the complete mean aggregate.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def binomial_pmf(trials: int, probability: float) -> np.ndarray:
    values = np.empty(trials + 1, dtype=np.float64)
    values[0] = (1.0 - probability) ** trials
    odds = probability / (1.0 - probability)
    for count in range(trials):
        values[count + 1] = (
            values[count] * (trials - count) / (count + 1) * odds
        )
    return values


def exact_ratio_information(
    graph_size: int, expected_degree: float, homophily: float
) -> dict[str, float | int]:
    same_candidates = graph_size // 2 - 1
    cross_candidates = graph_size // 2
    q_same = expected_degree * (1.0 + homophily) / (2.0 * same_candidates)
    q_cross = expected_degree * (1.0 - homophily) / (2.0 * cross_candidates)
    p_same = binomial_pmf(same_candidates, q_same)
    p_cross = binomial_pmf(cross_candidates, q_cross)

    # Each group stores [probability mass, probability-weighted rate score].
    groups: dict[bytes, list[float]] = defaultdict(lambda: [0.0, 0.0])
    raw_information = 0.0
    total_mass = 0.0
    for n_same, probability_same in enumerate(p_same):
        centered_same = n_same - same_candidates * q_same
        for n_cross, probability_cross in enumerate(p_cross):
            probability = probability_same * probability_cross
            if probability == 0.0:
                continue
            rate_score = (
                centered_same / (expected_degree * (1.0 - q_same))
                + (n_cross - cross_candidates * q_cross)
                / (expected_degree * (1.0 - q_cross))
            )
            total = n_same + n_cross
            if total == 0:
                key = b"empty"
            else:
                # Mirrors the float32 reduction/division in masked_mean.
                ratio = np.float32(np.float32(n_same - n_cross) / np.float32(total))
                key = ratio.tobytes()
            groups[key][0] += probability
            groups[key][1] += probability * rate_score
            raw_information += probability * rate_score * rate_score
            total_mass += probability

    ratio_information = sum(
        weighted_score * weighted_score / probability
        for probability, weighted_score in groups.values()
    )
    return {
        "graph_size": graph_size,
        "expected_degree": expected_degree,
        "homophily": homophily,
        "raw_rate_fisher": raw_information,
        "label_mean_rate_fisher": ratio_information,
        "label_mean_rate_efficiency": ratio_information / raw_information,
        "probability_mass": total_mass,
        "float32_ratio_groups": len(groups),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [
        exact_ratio_information(graph_size, expected_degree, 0.5)
        for graph_size in (512, 2048)
        for expected_degree in (16.0, 64.0)
    ]
    if args.output is None:
        for row in rows:
            print(row)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
