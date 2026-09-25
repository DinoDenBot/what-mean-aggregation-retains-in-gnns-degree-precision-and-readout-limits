import unittest

from smoothed_ratio_tv import smoothed_ratio_tv


class SmoothedRatioTVTests(unittest.TestCase):
    def test_composition_metric_isometry_in_three_categories(self) -> None:
        p = (0.2, 0.3, 0.5)
        contrast = ((1.0, -1.0, 0.0), (0.0, 1.0, -1.0))
        tangent = (0.2, 0.3, -0.5)
        sigma = [
            [
                (p[i] if i == j else 0.0) - p[i] * p[j]
                for j in range(3)
            ]
            for i in range(3)
        ]

        projected = [
            [
                sum(
                    contrast[a][i] * sigma[i][j] * contrast[b][j]
                    for i in range(3)
                    for j in range(3)
                )
                for b in range(2)
            ]
            for a in range(2)
        ]
        determinant = (
            projected[0][0] * projected[1][1]
            - projected[0][1] * projected[1][0]
        )
        inverse = (
            (projected[1][1] / determinant, -projected[0][1] / determinant),
            (-projected[1][0] / determinant, projected[0][0] / determinant),
        )
        transformed = tuple(
            sum(contrast[a][i] * tangent[i] for i in range(3))
            for a in range(2)
        )
        gaussian_metric = sum(
            transformed[i] * inverse[i][j] * transformed[j]
            for i in range(2)
            for j in range(2)
        )
        categorical_metric = sum(
            tangent[i] * tangent[i] / p[i] for i in range(3)
        )
        self.assertAlmostEqual(gaussian_metric, categorical_metric, places=12)

    def test_grid_preserves_probability_mass(self) -> None:
        result = smoothed_ratio_tv(50.0, bins_per_smoothing_sd=24)
        self.assertAlmostEqual(result.actual_grid_mass, 1.0, places=10)
        self.assertAlmostEqual(result.target_grid_mass, 1.0, places=10)
        self.assertLess(result.poisson_tail_upper_bound, 1e-12)

    def test_tv_decreases_in_reference_sequence(self) -> None:
        values = [
            smoothed_ratio_tv(rate, bins_per_smoothing_sd=24)
            .estimated_total_variation
            for rate in (25.0, 100.0, 400.0)
        ]
        self.assertGreater(values[0], values[1])
        self.assertGreater(values[1], values[2])
        self.assertLess(values[2], 0.02)

    def test_local_composition_shift_has_correct_gaussian_target(self) -> None:
        result = smoothed_ratio_tv(
            200.0,
            local_rate_shift=0.75,
            local_composition_shift=0.4,
            bins_per_smoothing_sd=24,
        )
        self.assertLess(result.estimated_total_variation, 0.01)

    def test_smoothing_exponent_must_lie_in_the_theorem_window(self) -> None:
        with self.assertRaises(ValueError):
            smoothed_ratio_tv(20.0, smoothing_exponent=0.5)
        with self.assertRaises(ValueError):
            smoothed_ratio_tv(20.0, smoothing_exponent=1.0)


if __name__ == "__main__":
    unittest.main()
