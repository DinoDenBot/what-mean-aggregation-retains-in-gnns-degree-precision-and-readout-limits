#!/usr/bin/env python3
"""Deterministic checks for two-dimensional Gaussian mean aggregation."""

from __future__ import annotations

import argparse
import math


def poisson_expectation(lam: float, fn) -> float:
    if lam <= 0:
        raise ValueError("lambda must be positive")
    radius = 16.0 * math.sqrt(lam) + 64.0
    lo = max(0, int(math.floor(lam - radius)))
    hi = int(math.ceil(lam + radius))
    total = 0.0
    mass = 0.0
    for n in range(lo, hi + 1):
        log_p = -lam + n * math.log(lam) - math.lgamma(n + 1)
        p = math.exp(log_p)
        total += p * fn(n)
        mass += p
    if mass < 1.0 - 1e-12:
        raise RuntimeError(f"insufficient Poisson support: retained mass={mass}")
    return total


def rate_efficiency(lam: float) -> float:
    return -math.expm1(-lam) / lam


def location_efficiency(lam: float) -> float:
    a_lam = poisson_expectation(lam, lambda n: n / (n + 1.0) ** 2)
    return 1.0 - a_lam


def posterior_mean_by_sum(lam: float, q: float) -> float:
    mu = lam * math.exp(-q / 2.0)
    hi = int(math.ceil(mu + 16.0 * math.sqrt(mu + 1.0) + 64.0))
    numerator = 0.0
    denominator = 0.0
    for n in range(1, hi + 1):
        log_w = n * math.log(mu) - math.lgamma(n)
        w = math.exp(log_w - mu)  # common scale is harmless in the ratio
        numerator += n * w
        denominator += w
    return numerator / denominator


def self_test() -> None:
    for lam, q in ((2.0, 0.4), (16.0, 0.1), (256.0, 0.01)):
        want = 1.0 + lam * math.exp(-q / 2.0)
        got = posterior_mean_by_sum(lam, q)
        assert abs(got - want) <= 2e-11 * max(1.0, want), (got, want)

    for lam in (1.0, 4.0, 16.0, 64.0, 256.0):
        # Total-variance calculation: Var(E[N|H]) = 1-exp(-lambda).
        expected_conditional_variance = lam * poisson_expectation(
            lam, lambda n: n / (n + 1.0)
        )
        assert abs((lam - expected_conditional_variance) - (-math.expm1(-lam))) < 2e-11

        # Direct integration of the pushed-forward rate score, conditional on N.
        offset = 1.0 / lam - 1.0
        direct_rate_fisher = math.exp(-lam) + poisson_expectation(
            lam,
            lambda n: 0.0 if n == 0 else (
                offset**2
                + 2.0 * offset * n / (n + 1.0)
                + n / (n + 2.0)
            ),
        )
        exact_rate_fisher = -math.expm1(-lam) / lam**2
        assert abs(direct_rate_fisher - exact_rate_fisher) < 2e-11

        # Direct location-score integration equals the missing-information form.
        direct_location_scalar = poisson_expectation(
            lam,
            lambda n: 0.0 if n == 0 else (
                1.0 / n
                + 2.0 * lam * n / (n + 1.0) ** 2
                + lam**2 * n / (n + 2.0) ** 2
            ),
        )
        a_lam = poisson_expectation(lam, lambda n: n / (n + 1.0) ** 2)
        exact_location_scalar = lam * (1.0 - a_lam)
        assert abs(direct_location_scalar - exact_location_scalar) < 3e-10

        eta_rate = rate_efficiency(lam)
        eta_location = location_efficiency(lam)
        assert 0.0 < eta_rate <= 1.0
        assert 0.0 < eta_location <= 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("gaussian-mean checks: PASS")
        return
    print("lambda,rate_efficiency,location_efficiency")
    for lam in (16.0, 32.0, 64.0, 128.0, 256.0):
        print(f"{lam:g},{rate_efficiency(lam):.12f},{location_efficiency(lam):.12f}")


if __name__ == "__main__":
    main()
