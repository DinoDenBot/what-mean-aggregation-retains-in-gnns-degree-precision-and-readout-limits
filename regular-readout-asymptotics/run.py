"""Population check of regular-readout and arithmetic-resolution asymptotics."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import poisson


ROOT = Path(__file__).resolve().parent


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def population(lam: float, p: np.ndarray, tail: float = 1e-14):
    maxima = []
    for mu in lam * p:
        upper = int(mu + 12 * np.sqrt(mu) + 30)
        while upper > 0 and poisson.sf(upper, mu) < tail / len(p):
            upper -= 1
        maxima.append(upper + 1)
    counts = np.stack(
        np.meshgrid(*[np.arange(upper + 1) for upper in maxima], indexing="ij"),
        axis=-1,
    ).reshape(-1, len(p))
    weights = np.exp(poisson.logpmf(counts, lam * p).sum(axis=1))
    live = weights > 0
    counts, weights = counts[live], weights[live]
    mass = weights.sum()
    weights /= mass
    degree = (counts.sum(axis=1) - lam) / np.sqrt(lam)
    degree -= weights @ degree
    raw = weights @ degree**2
    if abs(raw - 1) >= 1e-9:
        raise AssertionError((lam, p, "raw degree information", raw))
    return counts, weights, degree, {
        "mass": float(mass),
        "states": int(len(weights)),
        "maxima": maxima,
        "raw_degree_error": float(abs(raw - 1)),
    }


def scalar_projection(features: np.ndarray, weights: np.ndarray, score: np.ndarray) -> float:
    gram = features.T @ (weights[:, None] * features)
    cross = features.T @ (weights * score)
    return float(cross @ np.linalg.pinv(gram, rcond=1e-13) @ cross)


def grouped_information(keys: np.ndarray, weights: np.ndarray, score: np.ndarray) -> float:
    if keys.ndim == 1:
        _, inverse = np.unique(keys, return_inverse=True)
    else:
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    moment = np.bincount(inverse, weights=weights * score)
    return float(np.sum(moment[mass > 0] ** 2 / mass[mass > 0]))


def exact_keys(counts: np.ndarray) -> np.ndarray:
    divisor = np.gcd.reduce(counts, axis=1)
    return counts // np.maximum(divisor, 1)[:, None]


def binary_cell(lam: int, p1: float, tail: float = 1e-14):
    p = np.array([p1, 1 - p1])
    counts, weights, score, diagnostic = population(lam, p, tail)
    total = counts.sum(axis=1)
    ratio = counts[:, 0] / np.maximum(total, 1)
    x = np.zeros_like(ratio)
    nonempty = total > 0
    x[nonempty] = np.sqrt(lam) * (ratio[nonempty] - p1) / np.sqrt(p1 * (1 - p1))
    features = np.column_stack((np.ones(len(x)), x**2))
    quadratic = scalar_projection(features, weights, score)
    canonical = grouped_information(exact_keys(counts), weights, score)
    if quadratic > canonical + 1e-9:
        raise AssertionError((lam, p1, quadratic, canonical))
    base_bits = int(round(2 * math.log2(lam)))
    partitions = []
    for delta in (-4, -2, 0, 2):
        bits = max(0, base_bits + delta)
        keys = np.where(nonempty, np.floor(np.ldexp(ratio, bits)), -1)
        value = grouped_information(keys, weights, score)
        if value > canonical + 1e-9:
            raise AssertionError((lam, p1, bits, value, canonical))
        partitions.append((delta, bits, value))
    row = {
        "lambda": lam,
        "p1": p1,
        "canonical": canonical,
        "quadratic": quadratic,
        "prediction": 1 / (2 * lam),
        "lambda_times_quadratic": lam * quadratic,
        "quadratic_prediction_ratio": quadratic / (1 / (2 * lam)),
        "states": diagnostic["states"],
        "retained_mass": diagnostic["mass"],
    }
    for delta, bits, value in partitions:
        label = f"offset_{delta:+d}".replace("+", "plus").replace("-", "minus")
        row[f"{label}_bits"] = bits
        row[f"{label}_information"] = value
        row[f"{label}_canonical_fraction"] = value / canonical
    return row, diagnostic


def multicategory_cell(lam: int, p: np.ndarray, tail: float = 1e-14):
    counts, weights, score, diagnostic = population(lam, p, tail)
    total = counts.sum(axis=1)
    nonempty = total > 0
    ratios = counts / np.maximum(total, 1)[:, None]
    contrast = np.column_stack(
        [ratios[:, j] - ratios[:, -1] - (p[j] - p[-1]) for j in range(len(p) - 1)]
    )
    sigma = np.diag(p) - np.outer(p, p)
    basis = np.column_stack(
        [np.eye(len(p))[:, j] - np.eye(len(p))[:, -1] for j in range(len(p) - 1)]
    ).T
    covariance = basis @ sigma @ basis.T
    chol = np.linalg.cholesky(covariance)
    x = np.zeros_like(contrast)
    x[nonempty] = np.linalg.solve(chol, (np.sqrt(lam) * contrast[nonempty]).T).T
    q = np.sum(x**2, axis=1)
    quadratic = scalar_projection(np.column_stack((np.ones(len(q)), q)), weights, score)
    canonical = grouped_information(exact_keys(counts), weights, score)
    prediction = (len(p) - 1) / (2 * lam)
    if quadratic > canonical + 1e-9:
        raise AssertionError((lam, p, quadratic, canonical))
    return {
        "lambda": lam,
        "p": ",".join(map(str, p)),
        "dimension": len(p) - 1,
        "canonical": canonical,
        "quadratic": quadratic,
        "prediction": prediction,
        "lambda_times_quadratic": lam * quadratic,
        "quadratic_prediction_ratio": quadratic / prediction,
        "states": diagnostic["states"],
        "retained_mass": diagnostic["mass"],
    }, diagnostic


def main() -> None:
    started = time.monotonic()
    result = ROOT / "result"
    result.mkdir(exist_ok=True)
    sources = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "run.py", ROOT / "SPEC.md")
    }
    (result / "source.json").write_text(json.dumps(sources, indent=2) + "\n")

    binary_rows = []
    diagnostics = []
    for lam in (32, 64, 128, 256, 512, 1024):
        for p1 in (0.2, 0.35, 0.5):
            row, diagnostic = binary_cell(lam, p1)
            binary_rows.append(row)
            diagnostics.append({"family": "binary", "lambda": lam, "p": p1, **diagnostic})
            print("binary", lam, p1, row["quadratic"], row["canonical"], flush=True)
    write_csv(result / "binary.csv", binary_rows)

    multicategory_rows = []
    for lam in (16, 32, 64, 128):
        row, diagnostic = multicategory_cell(lam, np.array([0.2, 0.3, 0.5]))
        multicategory_rows.append(row)
        diagnostics.append({"family": "three-category", "lambda": lam, **diagnostic})
        print("three-category", lam, row["quadratic"], row["canonical"], flush=True)
    write_csv(result / "three_category.csv", multicategory_rows)

    cutoff_errors = []
    for lam, p1 in ((32, 0.2), (1024, 0.35), (1024, 0.5)):
        tight, _ = binary_cell(lam, p1, tail=1e-16)
        loose = next(row for row in binary_rows if row["lambda"] == lam and row["p1"] == p1)
        cutoff_errors.append(
            max(abs(tight[key] - loose[key]) for key in ("canonical", "quadratic"))
        )
    cutoff_max_error = max(cutoff_errors)
    if cutoff_max_error >= 1e-8:
        raise AssertionError(("cutoff stability", cutoff_max_error))

    validation = {
        "status": "passed",
        "binary_cells": len(binary_rows),
        "three_category_cells": len(multicategory_rows),
        "cutoff_max_error": cutoff_max_error,
        "diagnostics": diagnostics,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "elapsed_seconds": time.monotonic() - started,
    }
    (result / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    manifest_lines = []
    for path in sorted(result.iterdir()):
        if path.is_file() and path.name != "manifest.sha256":
            manifest_lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
    (result / "manifest.sha256").write_text("".join(manifest_lines))
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
