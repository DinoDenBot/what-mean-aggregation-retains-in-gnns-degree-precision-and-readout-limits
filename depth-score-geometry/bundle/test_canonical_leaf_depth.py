from __future__ import annotations

import itertools

from canonical_depth import (
    generalized_eigendirections,
    generalized_eigenvalues,
    matrix_relative_error,
    representation_fisher,
)
from canonical_leaf_depth import (
    compare_precision,
    eigendirection_certificates,
    leaf_laws_by_depth,
    quantized_mean_update,
    raw_leaf_fisher_by_depth,
    reciprocal_grid,
    score_leaf_fisher_by_depth,
)


def test_quantized_mean() -> None:
    assert quantized_mean_update(0, 99, 8) == 0
    assert quantized_mean_update(2, 1, 8) == 0
    assert quantized_mean_update(2, 3, 8) == 2
    assert reciprocal_grid(0.125) == 8


def test_small_leaf_tree() -> None:
    depths = [1, 2]
    raw = raw_leaf_fisher_by_depth(depths, 3, 1.0, 0.2, 0.0, 1.0)
    score = score_leaf_fisher_by_depth(depths, 3, 1.0, 0.2, 0.0, 1.0)
    laws = leaf_laws_by_depth(depths, 3, 1.0, 0.2, 4, 0.0, 1.0)
    for depth in depths:
        assert matrix_relative_error(score[depth], raw[depth]) < 1e-12
        law = laws[depth]
        assert abs(sum(mass[0] for mass in law.values()) - 1.0) < 1e-12
        assert abs(sum(mass[1] for mass in law.values())) < 1e-12
        assert abs(sum(mass[2] for mass in law.values())) < 1e-12
        fisher = representation_fisher(law, 0.0)
        eigenvalues = generalized_eigenvalues(fisher, raw[depth])
        assert max(eigenvalues) <= 1.0 + 1e-12
        directions = generalized_eigendirections(fisher, raw[depth], eigenvalues)
        residual, normalization = eigendirection_certificates(
            {
                "raw_fisher": [list(row) for row in raw[depth]],
                "representation_fisher": [list(row) for row in fisher],
                "generalized_eigenvalues": list(eigenvalues),
                "generalized_eigendirections": directions,
            }
        )
        assert residual < 1e-12
        assert normalization < 1e-12


def test_tiny_exhaustive_depth_one() -> None:
    candidates = 2
    lambda0 = 0.8
    theta0 = 0.2
    grid = 4
    q = lambda0 / candidates
    single = [
        (False, 0, 1.0 - q, (-1.0 / (candidates * (1.0 - q)), 0.0)),
        (
            True,
            -grid,
            q * (1.0 - theta0) / 2.0,
            (1.0 / (candidates * q), -1.0 / (1.0 - theta0)),
        ),
        (
            True,
            grid,
            q * (1.0 + theta0) / 2.0,
            (1.0 / (candidates * q), 1.0 / (1.0 + theta0)),
        ),
    ]
    direct = {}
    for outcomes in itertools.product(single, repeat=candidates):
        count = sum(int(item[0]) for item in outcomes)
        index_sum = sum(item[1] for item in outcomes)
        probability = 1.0
        score = [0.0, 0.0]
        for _, _, mass, contribution in outcomes:
            probability *= mass
            score[0] += contribution[0]
            score[1] += contribution[1]
        hidden = quantized_mean_update(count, index_sum, grid)
        value = direct.setdefault(hidden, [0.0, 0.0, 0.0])
        value[0] += probability
        value[1] += probability * score[0]
        value[2] += probability * score[1]
    recursive = leaf_laws_by_depth(
        [1], candidates, lambda0, theta0, grid, 0.0, 1.0
    )[1]
    assert set(direct) == set(recursive)
    for hidden, expected in direct.items():
        assert max(
            abs(actual - target)
            for actual, target in zip(recursive[hidden], expected)
        ) < 1e-12


def test_precision_comparison_rejects_ragged_matrices() -> None:
    valid = {
        "matrices": [
            {
                "delta": 0.5,
                "depth": 1,
                "grid": 2,
                "raw_fisher": [[1.0, 0.0], [0.0, 1.0]],
                "representation_fisher": [[1.0, 0.0], [0.0, 1.0]],
                "oracle_score_fisher": [[1.0, 0.0], [0.0, 1.0]],
            }
        ]
    }
    malformed = {
        "matrices": [
            {
                **valid["matrices"][0],
                "raw_fisher": [[1.0], [0.0, 1.0]],
            }
        ]
    }
    try:
        compare_precision(valid, malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("ragged precision matrix was accepted")


def main() -> None:
    test_quantized_mean()
    test_small_leaf_tree()
    test_tiny_exhaustive_depth_one()
    test_precision_comparison_rejects_ragged_matrices()
    print("canonical leaf-depth mechanical tests passed")


if __name__ == "__main__":
    main()
