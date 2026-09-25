"""Population precision audit and independent-data neural score probes."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import signal
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import poisson
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
BITS = [4, 6, 8, 10, 12, 16, 24]
SEEDS = [17, 29, 43]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scores(c, lam, p):
    n = c.sum(axis=1)
    comp = c[:, :-1] / p[:-1] - c[:, -1:] / p[-1]
    raw = lam * (np.diag(1 / p[:-1]) + np.ones((len(p)-1, len(p)-1)) / p[-1])
    chol = np.linalg.cholesky(raw)
    return np.column_stack(((n-lam)/np.sqrt(lam), np.linalg.solve(chol, comp.T).T))


def population(lam, p, tail=1e-14):
    p = np.asarray(p)
    # isf avoids catastrophic rounding of 1-tail for the tighter check.
    maxima = []
    for mu in lam*p:
        m = int(mu + 12*np.sqrt(mu) + 30)
        while poisson.sf(m, mu) < tail/len(p) and m > 0:
            m -= 1
        maxima.append(m+1)
    c = np.stack(np.meshgrid(*[np.arange(m+1) for m in maxima], indexing='ij'), -1).reshape(-1, len(p))
    w = np.exp(poisson.logpmf(c, lam*p).sum(axis=1))
    live = w > 0
    c, w = c[live], w[live]
    mass = w.sum()
    z = scores(c, lam, p)
    w /= mass
    mean = w @ z
    z -= mean  # score of the conditioned, truncated law
    raw = z.T @ (w[:, None]*z)
    assert abs(1-mass) < 1e-11
    assert np.max(np.abs(mean)) < 1e-10
    assert np.max(np.abs(raw-np.eye(len(p)))) < 1e-9
    return c, w, z, {'mass': float(mass), 'maxima': maxima,
                     'raw_error': float(np.max(np.abs(raw-np.eye(len(p))))),
                     'score_mean_error': float(np.max(np.abs(mean)))}


def group(keys, w, z):
    if keys.ndim == 1:
        _, inv = np.unique(keys, return_inverse=True)
    else:
        _, inv = np.unique(keys, axis=0, return_inverse=True)
    mass = np.bincount(inv, weights=w)
    moment = np.column_stack([np.bincount(inv, weights=w*z[:, j]) for j in range(z.shape[1])])
    live = mass > 0
    cond = np.zeros_like(moment)
    cond[live] = moment[live]/mass[live, None]
    fisher = cond.T @ (mass[:, None]*cond)
    return fisher, inv, cond


def check_psd(a, name, tol=1e-8):
    minimum = float(np.linalg.eigvalsh((a+a.T)/2).min())
    assert minimum >= -tol, (name, minimum)
    return minimum


def cast(x, dtype):
    if dtype == 'bfloat16':
        f = np.asarray(x, dtype=np.float32)
        u = f.view(np.uint32)
        rounded = (u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)
        return rounded.view(np.float32).astype(np.float64)
    return np.asarray(x).astype(dtype).astype(np.float64)


def arithmetic(c, dtype, pipeline):
    n = c.sum(axis=1)
    if pipeline == 'counts':
        out = cast(cast(c, dtype)/cast(np.maximum(n, 1), dtype)[:, None], dtype)
    else:
        out = cast(c/np.maximum(n, 1)[:, None], dtype)
    out[n == 0] = -1
    return out


def exact_keys(c):
    g = np.gcd.reduce(c, axis=1)
    return c//np.maximum(g, 1)[:, None]


def scalar_output(c, bits):
    n = c.sum(axis=1)
    x = c[:, 0]/np.maximum(n, 1)
    if bits != 'exact':
        x = np.floor(x*2**bits)/2**bits
    x[n == 0] = -1
    return x


def features(x):
    return np.column_stack((np.where(x < 0, 0, (x-.35)*np.sqrt(256/(.35*.65))), x < 0)).astype(np.float32)


def network():
    return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))


def train_probe(bits, seed, x_pop, w, z, canonical):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    c = rng.poisson(256*np.array([.35, .65]), size=(30000, 2))
    xx = torch.from_numpy(features(scalar_output(c, bits)))
    yy = torch.from_numpy(scores(c, 256, np.array([.35, .65])).astype(np.float32))
    model = network()
    opt = torch.optim.Adam(model.parameters(), lr=.001)
    best = math.inf
    state = None
    for step in range(800):
        indices = torch.from_numpy(rng.integers(0, 24000, size=1024))
        opt.zero_grad(set_to_none=True)
        loss = (model(xx[indices])-yy[indices]).square().mean()
        loss.backward()
        opt.step()
        if (step+1) % 50 == 0:
            with torch.no_grad():
                val = (model(xx[24000:])-yy[24000:]).square().mean().item()
            if val < best:
                best, best_step = val, step+1
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    torch.save(state, ROOT/'result'/f'probe-{bits}-{seed}.pt')
    with torch.no_grad():
        m = np.concatenate([model(torch.from_numpy(features(part))).numpy() for part in np.array_split(x_pop, max(1, len(x_pop)//8192))]).astype(float)
    value = m.T@(w[:, None]*z) + z.T@(w[:, None]*m) - m.T@(w[:, None]*m)
    raw = z.T@(w[:, None]*z)
    residual = (z-m).T@(w[:, None]*(z-m))
    assert np.max(np.abs(value-(raw-residual))) < 1e-10
    check_psd(canonical-value, 'probe lower bound')
    return {'bits': bits, 'seed': seed, 'degree': value[0, 0], 'composition': value[1, 1],
            'validation_mse': best, 'selected_step': best_step}


def primary(train=True, tail=1e-14):
    c, w, z, diagnostic = population(256, [.35, .65], tail)
    rows, partition_rows, probes = [], [], []
    for bits in BITS+['exact']:
        x = scalar_output(c, bits)
        keys = exact_keys(c) if bits == 'exact' else x
        fisher, inv, cond = group(keys, w, z)
        check_psd(np.eye(2)-fisher, 'raw contraction')
        missing = (z-cond[inv]).T@(w[:, None]*(z-cond[inv]))
        assert np.max(np.abs(fisher+missing-z.T@(w[:, None]*z))) < 1e-10
        row = {'bits': bits, 'degree': fisher[0, 0], 'composition': fisher[1, 1]}
        prev = np.zeros((2, 2))
        for b in [2, 4, 6, 8, 10, 12, 16, 24]:
            h = np.where(x < 0, -1, np.floor(x*2**b))
            pf, _, _ = group(h, w, z)
            check_psd(pf-prev, 'nested bins')
            check_psd(fisher-pf, 'partition contraction')
            prev = pf
            partition_rows.append({'bits': bits, 'decoder_bits': b, 'degree': pf[0, 0], 'composition': pf[1, 1]})
            if b in [8, 12, 16, 24]:
                row[f'partition{b}'] = pf[0, 0]
        if train:
            for seed in SEEDS:
                probes.append(train_probe(bits, seed, x, w, z, fisher))
                print('probe', bits, seed, probes[-1]['degree'], flush=True)
            vals = [r['degree'] for r in probes if r['bits'] == bits]
            row.update(probe_median=float(np.median(vals)), probe_min=min(vals), probe_max=max(vals))
        rows.append(row)
    return rows, partition_rows, probes, diagnostic


def sensitivity_cell(lam, p, tail=1e-14):
    c, w, z, diagnostic = population(lam, p, tail)
    exact, _, _ = group(exact_keys(c), w, z)
    rows = []
    for dtype in ['float32', 'float16', 'bfloat16']:
        for pipeline in ['output', 'counts']:
            out = arithmetic(c, dtype, pipeline)
            reduced, _, _ = group(out[:, :-1], w, z)
            full, _, _ = group(out, w, z)
            check_psd(full-reduced, 'reduced versus full')
            check_psd(np.eye(len(p))-full, 'arithmetic raw contraction')
            # Count casting need not be a coarsening of the exact ratio.
            if pipeline == 'output':
                check_psd(exact-full, 'output-only exact contraction')
            for view, f in [('reduced', reduced), ('full', full)]:
                rows.append({'lambda': lam, 'p': ','.join(map(str, p)), 'K': len(p),
                             'format': dtype, 'pipeline': pipeline, 'view': view,
                             'degree': f[0, 0], 'composition_min': np.linalg.eigvalsh(f[1:, 1:]).min(),
                             'exact_degree': exact[0, 0], 'states': len(w)})
    print('sensitivity', lam, p, 'states', len(w), flush=True)
    return rows, diagnostic


def finite_difference():
    lam, p = 64., np.array([.35, .65])
    c, w, z, _ = population(lam, p)
    f, inv, cond = group(arithmetic(c, 'bfloat16', 'counts'), w, z)
    mass = np.bincount(inv, weights=w)
    diffs = []
    for direction in range(2):
        h = 1e-4
        probs = []
        for sign in [-1, 1]:
            ls = lam+sign*h*np.sqrt(lam) if direction == 0 else lam
            ps = p.copy()
            if direction == 1:
                delta = sign*h*np.sqrt(p[0]*p[1]/lam)
                ps += np.array([delta, -delta])
            weights = np.exp(poisson.logpmf(c, ls*ps).sum(axis=1))
            weights /= weights.sum()
            probs.append(np.bincount(inv, weights=weights))
        derivative = (probs[1]-probs[0])/(2*h)
        live = mass > 1e-20
        diffs.append(float(np.sum((derivative[live]-mass[live]*cond[live, direction])**2/mass[live])))
    assert max(diffs) < 1e-12
    return diffs


def main():
    signal.alarm(7200)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    result = ROOT/'result'
    result.mkdir(exist_ok=True)
    started = time.monotonic()
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'run.py', ROOT/'SPEC.md']}
    (result/'source.json').write_text(json.dumps(source, indent=2)+'\n')
    test_values = np.concatenate((np.arange(4097), np.linspace(0, 1, 5001), np.array([.35, .65])))
    for dtype, td in [('float32', torch.float32), ('float16', torch.float16), ('bfloat16', torch.bfloat16)]:
        expected = torch.from_numpy(test_values.astype(np.float32)).to(td).double().numpy()
        assert np.array_equal(cast(test_values.astype(np.float32), dtype), expected)
    rows, partitions, probes, diagnostic = primary()
    write_csv(result/'primary.csv', rows)
    write_csv(result/'partitions.csv', partitions)
    write_csv(result/'probes.csv', probes)
    tighter, _, _, _ = primary(train=False, tail=1e-16)
    cutoff_error = max(abs(a[k]-b[k]) for a, b in zip(rows, tighter) for k in ['degree', 'composition'])
    assert cutoff_error < 1e-8
    cells = [(l, [p, 1-p]) for l in [64, 256, 1024] for p in [.1, .35, .5]]
    cells += [(l, p) for l in [16, 64] for p in [[.2, .3, .5], [.05, .15, .8]]]
    sensitivity, diagnostics = [], []
    for lam, p in cells:
        rr, dd = sensitivity_cell(lam, p)
        sensitivity.extend(rr)
        diagnostics.append(dd)
        write_csv(result/'arithmetic.csv', sensitivity)
    for lam, p in [(1024, [.35, .65]), (64, [.2, .3, .5])]:
        rr, _ = sensitivity_cell(lam, p, 1e-16)
        previous = [r for r in sensitivity if r['lambda'] == lam and r['p'] == ','.join(map(str, p))]
        err = max(abs(a[k]-b[k]) for a, b in zip(rr, previous) for k in ['degree', 'composition_min'])
        cutoff_error = max(cutoff_error, err)
        assert err < 1e-8
    reference = [r for r in sensitivity if r['lambda'] == 1024 and r['p'] == '0.35,0.65' and r['pipeline'] == 'counts' and r['view'] == 'reduced']
    expected = {'float32': .9185, 'float16': .00459, 'bfloat16': .00233}
    for r in reference:
        assert abs(r['degree']-expected[r['format']]) < 5e-5
    validation = {'status': 'passed', 'primary': diagnostic, 'sensitivity': diagnostics,
                  'cutoff_max_error': cutoff_error, 'finite_difference_weighted_errors': finite_difference(),
                  'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                  'torch': torch.__version__, 'threads': 4, 'elapsed_seconds': time.monotonic()-started,
                  'primary_channels': len(rows), 'probe_fits': len(probes), 'arithmetic_rows': len(sensitivity)}
    (result/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    (result/'manifest.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n' for p in sorted(result.iterdir()) if p.is_file() and p.name != 'manifest.sha256'))
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == '__main__':
    main()
