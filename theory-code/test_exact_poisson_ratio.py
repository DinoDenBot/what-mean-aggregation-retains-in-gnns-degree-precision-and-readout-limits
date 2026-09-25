import math
import unittest

from exact_poisson_ratio import exact_ratio_fisher


class ExactPoissonRatioTests(unittest.TestCase):
    def test_whitened_raw_fisher_is_identity(self) -> None:
        result = exact_ratio_fisher(12.0, 0.37, relative_tail=1e-15)
        self.assertAlmostEqual(result.raw_rate_check, 1.0, places=11)
        self.assertAlmostEqual(result.raw_composition_check, 1.0, places=11)
        self.assertAlmostEqual(result.raw_cross_check, 0.0, places=11)

    def test_data_processing(self) -> None:
        result = exact_ratio_fisher(20.0, 0.5)
        self.assertGreaterEqual(result.eigenvalues[0], -1e-12)
        self.assertLessEqual(result.eigenvalues[1], 1 + 1e-10)

    def test_empty_symbol_contribution_at_tiny_rate(self) -> None:
        result = exact_ratio_fisher(0.2, 0.5)
        expected_lower_bound = 0.2 / math.expm1(0.2)
        self.assertGreaterEqual(result.rate, expected_lower_bound - 1e-12)

    def test_quantization_cannot_increase_fisher_information(self) -> None:
        exact = exact_ratio_fisher(30.0, 0.4)
        quantized = exact_ratio_fisher(30.0, 0.4, quantization_width=0.05)
        self.assertLessEqual(quantized.eigenvalues[1], exact.eigenvalues[1] + 1e-10)
        self.assertLessEqual(quantized.rate, exact.rate + 1e-10)
        self.assertLessEqual(quantized.composition, exact.composition + 1e-10)


if __name__ == "__main__":
    unittest.main()
