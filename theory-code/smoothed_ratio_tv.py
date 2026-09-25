#!/usr/bin/env python3
"""Numerical TV check for the two-category robust-ratio theorem.

For contrast L=(1,-1), the standardized observation is

    Y = 2 * sqrt(lambda0) * (C1 / N - p0) + sqrt(lambda0) * tau * Z,

with a fixed output for N=0.  The exact discrete ratio law is convolved with
the Gaussian channel on a fine one-dimensional grid and compared with the
limiting N(2u, 4 p0 (1-p0)) density.

This is a deterministic quadrature check, not part of the proof.  It uses only
the Python standard library and reports its Poisson truncation and grid
settings explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass

from exact_poisson_ratio import _binomial_probabilities, _poisson_probabilities


@dataclass(frozen=True)
class SmoothedTVResult:
    nominal_rate: float
    data_rate: float
    p0: float
    data_p: float
    local_rate_shift: float
    local_composition_shift: float
    smoothing_exponent: float
    raw_smoothing_width: float
    standardized_smoothing_sd: float
    estimated_total_variation: float
    actual_grid_mass: float
    target_grid_mass: float
    poisson_tail_upper_bound: float
    mean_binning_tv_upper_bound: float
    gaussian_kernel_tail_upper_bound: float
    bins_per_smoothing_sd: int
    grid_points: int
    mixture_bins: int
    n_max: int


def _normal_density(x: float, mean: float, sd: float) -> float:
    z = (x - mean) / sd
    return math.exp(-0.5 * z * z) / (math.sqrt(2 * math.pi) * sd)


def smoothed_ratio_tv(
    nominal_rate: float,
    *,
    p0: float = 0.5,
    local_rate_shift: float = 0.0,
    local_composition_shift: float = 0.0,
    smoothing_exponent: float = 0.75,
    relative_tail: float = 1e-14,
    bins_per_smoothing_sd: int = 32,
    gaussian_tail_sd: float = 8.0,
    target_tail_sd: float = 9.0,
    empty_contrast: float = 0.0,
) -> SmoothedTVResult:
    if nominal_rate <= 0 or not math.isfinite(nominal_rate):
        raise ValueError("nominal_rate must be finite and positive")
    if not 0 < p0 < 1:
        raise ValueError("p0 must lie in (0, 1)")
    if not 0.5 < smoothing_exponent < 1:
        raise ValueError("smoothing_exponent must lie in (1/2, 1)")
    if bins_per_smoothing_sd < 8:
        raise ValueError("bins_per_smoothing_sd must be at least 8")

    root_rate = math.sqrt(nominal_rate)
    data_rate = nominal_rate + local_rate_shift * root_rate
    data_p = p0 + local_composition_shift / root_rate
    if data_rate <= 0 or not 0 < data_p < 1:
        raise ValueError("local alternative leaves the parameter domain")

    tau = nominal_rate ** (-smoothing_exponent)
    smoothing_sd = root_rate * tau
    target_mean = 2 * local_composition_shift
    target_sd = 2 * math.sqrt(p0 * (1 - p0))
    dx = smoothing_sd / bins_per_smoothing_sd

    minimum_ratio_mean = -2 * p0 * root_rate
    maximum_ratio_mean = 2 * (1 - p0) * root_rate
    empty_mean = root_rate * (empty_contrast - (2 * p0 - 1))
    x_min = min(
        minimum_ratio_mean,
        empty_mean,
        target_mean - target_tail_sd * target_sd,
    ) - gaussian_tail_sd * smoothing_sd
    x_max = max(
        maximum_ratio_mean,
        empty_mean,
        target_mean + target_tail_sd * target_sd,
    ) + gaussian_tail_sd * smoothing_sd
    grid_points = int(math.ceil((x_max - x_min) / dx))
    x_max = x_min + grid_points * dx

    poisson, poisson_tail = _poisson_probabilities(data_rate, relative_tail)
    mixture_weights = [0.0] * grid_points

    def add_component(mean: float, weight: float) -> None:
        index = int(math.floor((mean - x_min) / dx))
        if index < 0:
            index = 0
        elif index >= grid_points:
            index = grid_points - 1
        mixture_weights[index] += weight

    for n, probability_n in enumerate(poisson):
        if probability_n == 0:
            continue
        if n == 0:
            add_component(empty_mean, probability_n)
            continue
        for c1, probability_c1 in enumerate(_binomial_probabilities(n, data_p)):
            if probability_c1:
                mean = 2 * root_rate * (c1 / n - p0)
                add_component(mean, probability_n * probability_c1)

    kernel_radius = int(math.ceil(gaussian_tail_sd * bins_per_smoothing_sd))
    kernel = [
        _normal_density(offset * dx, 0.0, smoothing_sd)
        for offset in range(-kernel_radius, kernel_radius + 1)
    ]
    kernel_mass = math.fsum(kernel) * dx
    kernel = [value / kernel_mass for value in kernel]

    actual_density = [0.0] * grid_points
    occupied = 0
    for source_index, weight in enumerate(mixture_weights):
        if weight == 0:
            continue
        occupied += 1
        lower = max(0, source_index - kernel_radius)
        upper = min(grid_points - 1, source_index + kernel_radius)
        for target_index in range(lower, upper + 1):
            kernel_index = target_index - source_index + kernel_radius
            actual_density[target_index] += weight * kernel[kernel_index]

    target_density = []
    for index in range(grid_points):
        x = x_min + (index + 0.5) * dx
        target_density.append(_normal_density(x, target_mean, target_sd))

    actual_mass = math.fsum(actual_density) * dx
    target_mass = math.fsum(target_density) * dx
    tv = 0.5 * math.fsum(
        abs(actual - target)
        for actual, target in zip(actual_density, target_density)
    ) * dx

    return SmoothedTVResult(
        nominal_rate=nominal_rate,
        data_rate=data_rate,
        p0=p0,
        data_p=data_p,
        local_rate_shift=local_rate_shift,
        local_composition_shift=local_composition_shift,
        smoothing_exponent=smoothing_exponent,
        raw_smoothing_width=tau,
        standardized_smoothing_sd=smoothing_sd,
        estimated_total_variation=tv,
        actual_grid_mass=actual_mass,
        target_grid_mass=target_mass,
        poisson_tail_upper_bound=poisson_tail,
        mean_binning_tv_upper_bound=1 / (4 * bins_per_smoothing_sd),
        gaussian_kernel_tail_upper_bound=math.erfc(
            gaussian_tail_sd / math.sqrt(2)
        ),
        bins_per_smoothing_sd=bins_per_smoothing_sd,
        grid_points=grid_points,
        mixture_bins=occupied,
        n_max=len(poisson) - 1,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", default="25,100,400")
    parser.add_argument("--p0", type=float, default=0.5)
    parser.add_argument("--h", type=float, default=0.0)
    parser.add_argument("--u", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--relative-tail", type=float, default=1e-14)
    parser.add_argument("--bins-per-sd", type=int, default=32)
    args = parser.parse_args()

    results = []
    for value in args.rates.split(","):
        result = smoothed_ratio_tv(
            float(value),
            p0=args.p0,
            local_rate_shift=args.h,
            local_composition_shift=args.u,
            smoothing_exponent=args.alpha,
            relative_tail=args.relative_tail,
            bins_per_smoothing_sd=args.bins_per_sd,
        )
        results.append(asdict(result))
    print(json.dumps({"schema_version": 1, "results": results}, indent=2))


if __name__ == "__main__":
    main()

