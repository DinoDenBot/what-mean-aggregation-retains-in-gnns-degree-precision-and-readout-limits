from __future__ import annotations

from canonical_depth import (
    generalized_eigenvalues,
    hidden_laws_by_depth,
    matrix_relative_error,
    raw_fisher_by_depth,
    representation_fisher,
    score_moment_fisher_by_depth,
    quantized_update,
    round_ratio_ties_even,
)


def test_rounding() -> None:
    assert round_ratio_ties_even(1, 2) == 0
    assert round_ratio_ties_even(3, 2) == 2
    assert round_ratio_ties_even(-1, 2) == 0
    assert round_ratio_ties_even(-3, 2) == -2


def test_update_range_and_empty_convention() -> None:
    assert quantized_update(1, 0, 0, 16) == 8
    assert quantized_update(-1, 0, 0, 16) == -8
    assert quantized_update(1, 4, -64, 16) == 0
    assert quantized_update(-1, 4, 64, 16) == 0


def test_small_finite_tree_identities() -> None:
    depths = [1, 2]
    candidates = 2
    raw = raw_fisher_by_depth(depths, candidates, 0.5, 0.2, 0.0, 1.0)
    score = score_moment_fisher_by_depth(
        depths, candidates, 0.5, 0.2, 0.0, 1.0
    )
    laws = hidden_laws_by_depth(
        depths, candidates, 0.5, 0.2, 4, 0.0, 1.0
    )
    for depth in depths:
        assert matrix_relative_error(score[depth][1], raw[depth][1]) < 1e-12
        law = laws[depth][1]
        assert abs(sum(mass[0] for mass in law.values()) - 1.0) < 1e-12
        assert abs(sum(mass[1] for mass in law.values())) < 1e-12
        assert abs(sum(mass[2] for mass in law.values())) < 1e-12
        fisher = representation_fisher(law, 0.0)
        assert max(generalized_eigenvalues(fisher, raw[depth][1])) <= 1.0 + 1e-12


def main() -> None:
    test_rounding()
    test_update_range_and_empty_convention()
    test_small_finite_tree_identities()
    print("canonical-depth mechanical tests passed")


if __name__ == "__main__":
    main()
