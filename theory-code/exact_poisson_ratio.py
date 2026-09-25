#!/usr/bin/env python3
"""Exact truncated Fisher calculation for a two-type Poisson ratio statistic.

The model is

    N ~ Poisson(lambda),  C1 | N ~ Binomial(N, p),  C2 = N - C1.

The statistic is the primitive ratio (C1, C2) / (C1 + C2), with a separate
symbol for N = 0.  Scores are expressed in raw-Fisher-orthonormal rate and
composition coordinates, so the uncompressed Fisher matrix is the identity.

Only the Python standard library is used.  The Poisson total is truncated with
an explicit relative tail bound; every conditional binomial law is enumerated
in full.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Union


RatioKey = Union[Tuple[int, int], str]


@dataclass(frozen=True)
class FisherResult:
    rate: float
    composition: float
    cross: float
    eigenvalues: Tuple[float, float]
    included_probability_lower_bound: float
    score_mean_rate: float
    score_mean_composition: float
    raw_rate_check: float
    raw_composition_check: float
    raw_cross_check: float
    groups: int
    n_max: int
    channel: str

    def as_dict(self) -> dict[str, object]:
        return {
            "retained_fisher_whitened": [
                [self.rate, self.cross],
                [self.cross, self.composition],
            ],
            "generalized_eigenvalues": list(self.eigenvalues),
            "included_probability_lower_bound": self.included_probability_lower_bound,
            "score_mean": [self.score_mean_rate, self.score_mean_composition],
            "raw_fisher_check": [
                [self.raw_rate_check, self.raw_cross_check],
                [self.raw_cross_check, self.raw_composition_check],
            ],
            "groups": self.groups,
            "n_max": self.n_max,
            "channel": self.channel,
        }


def _poisson_probabilities(rate: float, relative_tail: float) -> tuple[list[float], float]:
    """Return normalized probabilities from zero to n_max and a tail bound.

    We work relative to the Poisson mode to avoid exp(-rate) underflow.  Once
    the upper tail is geometrically bounded by ``relative_tail`` times the
    accumulated mass, normalization gives an absolute omitted-mass bound.
    """

    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("rate must be finite and positive")
    if not 0 < relative_tail < 1:
        raise ValueError("relative_tail must lie in (0, 1)")

    mode = int(math.floor(rate))
    weights = [0.0] * (mode + 1)
    weights[mode] = 1.0
    for n in range(mode, 0, -1):
        weights[n - 1] = weights[n] * n / rate

    total = math.fsum(weights)
    n = mode
    weight = 1.0
    tail_bound = math.inf
    while True:
        n += 1
        weight *= rate / n
        weights.append(weight)
        total += weight
        next_weight = weight * rate / (n + 1)
        ratio_bound = rate / (n + 2)
        if ratio_bound < 1:
            tail_bound = next_weight / (1 - ratio_bound)
            if tail_bound <= relative_tail * total:
                break

    mode_probability = math.exp(
        -rate + mode * math.log(rate) - math.lgamma(mode + 1)
    )
    probabilities = [weight * mode_probability for weight in weights]
    omitted_upper_bound = tail_bound * mode_probability
    return probabilities, omitted_upper_bound


def _binomial_probabilities(n: int, p: float) -> list[float]:
    """Enumerate Binomial(n, p) stably using a recurrence around its mode."""

    if not 0 < p < 1:
        raise ValueError("p must lie in (0, 1)")
    if n == 0:
        return [1.0]

    mode = min(n, int(math.floor((n + 1) * p)))
    weights = [0.0] * (n + 1)
    weights[mode] = 1.0
    odds = p / (1 - p)

    for k in range(mode, n):
        weights[k + 1] = weights[k] * (n - k) / (k + 1) * odds
    for k in range(mode, 0, -1):
        weights[k - 1] = weights[k] * k / (n - k + 1) / odds

    total = math.fsum(weights)
    return [weight / total for weight in weights]


def _eigenvalues_2x2(a: float, b: float, c: float) -> tuple[float, float]:
    trace = a + c
    radius = math.hypot(a - c, 2 * b)
    return ((trace - radius) / 2, (trace + radius) / 2)


def exact_ratio_fisher(
    rate: float,
    p: float,
    *,
    relative_tail: float = 1e-14,
    quantization_width: float | None = None,
) -> FisherResult:
    """Compute retained Fisher information for an exact or quantized ratio.

    A positive ``quantization_width`` rounds C1 / N to a fixed grid.  The grid
    is frozen for the call and therefore defines a parameter-independent
    channel around the supplied nominal rate.
    """

    if quantization_width is not None and quantization_width <= 0:
        raise ValueError("quantization_width must be positive when supplied")

    poisson, tail_bound = _poisson_probabilities(rate, relative_tail)
    groups: Dict[RatioKey, list[float]] = {}

    mean_rate = 0.0
    mean_comp = 0.0
    raw_rate = 0.0
    raw_comp = 0.0
    raw_cross = 0.0
    rate_scale = math.sqrt(rate)
    comp_scale = math.sqrt(rate * p * (1 - p))

    for n, p_n in enumerate(poisson):
        if p_n == 0:
            continue
        binomial = _binomial_probabilities(n, p)
        for c1, p_c_given_n in enumerate(binomial):
            mass = p_n * p_c_given_n
            if mass == 0:
                continue
            c2 = n - c1
            if n == 0:
                key: RatioKey = "empty"
            elif quantization_width is not None:
                key = (int(math.floor((c1 / n) / quantization_width + 0.5)), -1)
            else:
                divisor = math.gcd(c1, c2)
                key = (c1 // divisor, c2 // divisor)

            z_rate = (n - rate) / rate_scale
            z_comp = (c1 - n * p) / comp_scale
            bucket = groups.setdefault(key, [0.0, 0.0, 0.0])
            bucket[0] += mass
            bucket[1] += mass * z_rate
            bucket[2] += mass * z_comp

            mean_rate += mass * z_rate
            mean_comp += mass * z_comp
            raw_rate += mass * z_rate * z_rate
            raw_comp += mass * z_comp * z_comp
            raw_cross += mass * z_rate * z_comp

    retained_rate = 0.0
    retained_comp = 0.0
    retained_cross = 0.0
    for probability, weighted_rate, weighted_comp in groups.values():
        retained_rate += weighted_rate * weighted_rate / probability
        retained_comp += weighted_comp * weighted_comp / probability
        retained_cross += weighted_rate * weighted_comp / probability

    eigenvalues = _eigenvalues_2x2(retained_rate, retained_cross, retained_comp)
    return FisherResult(
        rate=retained_rate,
        composition=retained_comp,
        cross=retained_cross,
        eigenvalues=eigenvalues,
        included_probability_lower_bound=1 - tail_bound,
        score_mean_rate=mean_rate,
        score_mean_composition=mean_comp,
        raw_rate_check=raw_rate,
        raw_composition_check=raw_comp,
        raw_cross_check=raw_cross,
        groups=len(groups),
        n_max=len(poisson) - 1,
        channel=(
            "exact_ratio"
            if quantization_width is None
            else f"quantized_ratio:{quantization_width:.17g}"
        ),
    )


def _parse_rates(text: str) -> Iterable[float]:
    for value in text.split(","):
        yield float(value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", default="10,25,50,100,250,500,1000")
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument("--relative-tail", type=float, default=1e-14)
    parser.add_argument("--quantization-width", type=float)
    parser.add_argument(
        "--quantization-exponent",
        type=float,
        help="Use a nominal-rate grid width rate**(-exponent).",
    )
    args = parser.parse_args()

    rows = []
    for rate in _parse_rates(args.rates):
        if args.quantization_width is not None and args.quantization_exponent is not None:
            parser.error("choose only one quantization option")
        width = args.quantization_width
        if args.quantization_exponent is not None:
            width = rate ** (-args.quantization_exponent)
        result = exact_ratio_fisher(
            rate,
            args.p,
            relative_tail=args.relative_tail,
            quantization_width=width,
        )
        rows.append({"rate": rate, "p": args.p, **result.as_dict()})
    print(json.dumps({"schema_version": 1, "results": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
