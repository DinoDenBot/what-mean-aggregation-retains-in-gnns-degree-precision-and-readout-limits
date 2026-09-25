"""Independent moment verification for the binary quadratic decoder."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import poisson


ROOT = Path(__file__).resolve().parent


def analytic_bound(lam: float, p: float, tail: float = 1e-16) -> float:
    upper = int(lam + 14 * math.sqrt(lam) + 50)
    n = np.arange(1, upper + 1)
    weights = poisson.pmf(n, lam)
    empty = math.exp(-lam)
    ex2 = float(np.sum(weights * lam / n))
    ex4 = float(
        np.sum(
            weights
            * lam**2
            * (3 / n**2 + (1 - 6 * p * (1 - p)) / (p * (1 - p) * n**3))
        )
    )
    ez = (lam * (1 - empty) - lam) / math.sqrt(lam)
    ez2 = float(np.sum(weights * ((n - lam) / math.sqrt(lam)) ** 2) + empty * lam)
    centered_scale = math.sqrt(ez2 - ez**2)
    score = ((n - lam) / math.sqrt(lam) - ez) / centered_scale
    cross = float(np.sum(weights * score * lam / n))
    variance = ex4 - ex2**2
    if poisson.sf(upper, lam) >= tail:
        raise AssertionError((lam, upper, poisson.sf(upper, lam)))
    return cross**2 / variance


def main() -> None:
    with (ROOT / "result" / "binary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    for row in rows:
        expected = analytic_bound(float(row["lambda"]), float(row["p1"]))
        errors.append(abs(expected - float(row["quadratic"])))
    maximum = max(errors)
    if maximum >= 1e-10:
        raise AssertionError(maximum)
    result = {
        "status": "passed",
        "cells": len(rows),
        "max_absolute_error": maximum,
        "method": "one-dimensional Poisson sum with conditional binomial moments",
    }
    (ROOT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
