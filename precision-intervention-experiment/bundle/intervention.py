from __future__ import annotations

import math
import torch


def perturb(mean, degree, channel, layer, noise=None):
    if layer not in channel['layers'] or channel['kind'] == 'native':
        return mean
    kind = channel['kind']
    if kind in ('float16', 'bfloat16'):
        return mean.to(getattr(torch, kind)).to(mean.dtype)
    if layer != 1 or mean.shape[1] != 8 or noise is None:
        raise ValueError('Additive noise is defined only for the eight first-layer coordinates')
    sigma = torch.ones(8, dtype=mean.dtype, device=mean.device)
    sigma[0] = math.sqrt(1 + .35 ** 2)
    sigma *= 24 ** (-.75) * channel['scale']
    delta = noise * sigma
    return mean + delta * (degree > 0).unsqueeze(1)


def aggregate(layer, x, edge_src, degree, channel, index, noise=None):
    # Same mean reduction as the original code; unused min/max/sum are omitted.
    mean = torch.nan_to_num(torch.segment_reduce(x[edge_src], 'mean', lengths=degree))
    changed = perturb(mean, degree, channel, index, noise)
    delta_square = float((changed - mean).double().square().mean())
    mean_square = float(mean.double().square().mean())
    if layer.architecture == 'mean_degree':
        standardized = (torch.log1p(degree.to(x.dtype)) - layer.degree_location) / layer.degree_scale
        changed = torch.cat([changed, standardized.unsqueeze(1)], dim=1)
    elif layer.architecture != 'mean':
        raise ValueError(layer.architecture)
    return changed, delta_square, mean_square


def forward(model, graph, channel, noise=None):
    hidden = graph.x
    diagnostics = {}
    for index, layer in enumerate((model.layer1, model.layer2), 1):
        agg, delta2, signal2 = aggregate(layer, hidden, graph.edge_src, graph.degree, channel, index, noise)
        hidden = layer.dropout(layer.activation(layer.affine(torch.cat([hidden, agg], dim=1))))
        diagnostics[f'layer{index}_noise_rms'] = math.sqrt(delta2)
        diagnostics[f'layer{index}_mean_rms'] = math.sqrt(signal2)
    return model.classifier(hidden).squeeze(-1), diagnostics
