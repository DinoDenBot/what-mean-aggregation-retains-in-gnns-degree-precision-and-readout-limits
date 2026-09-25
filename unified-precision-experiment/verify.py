"""Independent total-count/binomial enumeration and checkpoint evaluation.

The production run enumerates independent Poisson marginals. This validator
instead conditions on their total and enumerates a binomial composition.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import poisson, binom
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent


def read(name):
    return list(csv.DictReader((ROOT/'result'/name).open()))


def support(lam):
    total, first, mass = [], [], []
    for n in range(int(lam+12*np.sqrt(lam)+50)+1):
        k = np.arange(n+1)
        total.append(np.full(n+1, n))
        first.append(k)
        mass.append(poisson.pmf(n, lam)*binom.pmf(k, n, .35))
    n, k, w = map(np.concatenate, (total, first, mass))
    live = w > 0
    n, k, w = n[live], k[live], w[live]
    w /= w.sum()
    z = np.column_stack(((n-lam)/np.sqrt(lam), (k-.35*n)/np.sqrt(lam*.35*.65)))
    return n, k, w, z


def information(key, w, z):
    _, inv = np.unique(key, axis=0, return_inverse=True)
    mass = np.bincount(inv, weights=w)
    moment = np.column_stack([np.bincount(inv, weights=w*s) for s in z.T])
    live = mass > 0
    m = moment[live]/mass[live, None]
    return (mass[live, None]*m*m).sum(axis=0)


def verify():
    torch.set_num_threads(4)
    for line in (ROOT/'result/manifest.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        assert hashlib.sha256((ROOT/'result'/name).read_bytes()).hexdigest() == digest
    for name, digest in json.loads((ROOT/'result/source.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    n, k, w, z = support(256)
    ratio = k/np.maximum(n, 1)
    primary_errors, probe_errors, partition_errors = [], [], []
    for row in read('primary.csv'):
        b = row['bits']
        x = ratio.copy() if b == 'exact' else np.floor(ratio*2**int(b))/2**int(b)
        x[n == 0] = -1
        if b == 'exact':
            g = np.maximum(np.gcd(k, n), 1)
            key = np.column_stack((k//g, n//g))
        else:
            key = x[:, None]
        f = information(key, w, z)
        primary_errors.extend(abs(f[i]-float(row[col])) for i, col in enumerate(['degree', 'composition']))
        for part in [r for r in read('partitions.csv') if r['bits'] == b]:
            key = np.where(x < 0, -1, np.floor(x*2**int(part['decoder_bits'])))[:, None]
            f = information(key, w, z)
            partition_errors.extend(abs(f[i]-float(part[col])) for i, col in enumerate(['degree', 'composition']))
        features = np.column_stack((np.where(x < 0, 0, (x-.35)*np.sqrt(256/(.35*.65))), x < 0)).astype(np.float32)
        for pr in [r for r in read('probes.csv') if r['bits'] == b]:
            model = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
            model.load_state_dict(torch.load(ROOT/'result'/f"probe-{b}-{pr['seed']}.pt", weights_only=True))
            with torch.no_grad():
                m = np.concatenate([model(torch.from_numpy(a)).numpy() for a in np.array_split(features, 16)]).astype(float)
            # Independent evaluation as raw score variance minus prediction MSE.
            achieved = (w[:, None]*(z*z-(z-m)**2)).sum(axis=0)
            probe_errors.extend(abs(achieved[i]-float(pr[col])) for i, col in enumerate(['degree', 'composition']))
    n, k, w, z = support(1024)
    c = np.column_stack((k, n-k))
    arithmetic_errors = []
    for row in read('arithmetic.csv'):
        if row['lambda'] != '1024' or row['p'] != '0.35,0.65':
            continue
        dtype = getattr(torch, row['format'])
        if row['pipeline'] == 'counts':
            numerator = torch.from_numpy(c.astype(np.float32)).to(dtype).double()
            denominator = torch.from_numpy(np.maximum(n, 1).astype(np.float32)).to(dtype).double()
            out = (numerator/denominator[:, None]).to(dtype).double().numpy()
        else:
            out = torch.from_numpy(c/np.maximum(n, 1)[:, None]).to(dtype).double().numpy()
        out[n == 0] = -1
        if row['view'] == 'reduced':
            out = out[:, :1]
        f = information(out, w, z)
        arithmetic_errors.extend([abs(f[0]-float(row['degree'])), abs(f[1]-float(row['composition_min']))])
    result = dict(status='passed', independent_enumeration='Poisson total and conditional binomial',
                  primary_max_error=float(max(primary_errors)), partition_max_error=float(max(partition_errors)),
                  checkpoint_score_max_error=float(max(probe_errors)),
                  arithmetic_torch_max_error=float(max(arithmetic_errors)))
    assert max(primary_errors+partition_errors+probe_errors+arithmetic_errors) < 1e-8, result
    (ROOT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
