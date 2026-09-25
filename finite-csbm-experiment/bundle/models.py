from __future__ import annotations

import math

import torch
from torch import nn


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    count = mask.sum(1, keepdim=True).clamp_min(1).to(x.dtype)
    return (x * mask.unsqueeze(-1)).sum(1) / count


def empty_indicator(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (mask.sum(1) == 0).to(dtype).unsqueeze(-1)


class LocalAggregator(nn.Module):
    """A frozen one-hop aggregation channel plus a common representation trunk."""

    def __init__(self, name: str, lambda0: float) -> None:
        super().__init__()
        self.name = name
        self.lambda0 = float(lambda0)
        self.message_dim = 4
        if name == "gat":
            self.key = nn.Linear(self.message_dim, 16, bias=False)
            self.query = nn.Parameter(torch.zeros(16))
            nn.init.normal_(self.query, std=0.05)
        aggregate_dims = {
            "mean": 5,
            "sum": 5,
            "mean_degree": 6,
            "gat": 5,
        }
        if name not in aggregate_dims:
            raise ValueError(f"unknown architecture: {name}")
        self.aggregate_dim = aggregate_dims[name]
        self.trunk = nn.Sequential(
            nn.Linear(self.aggregate_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 16),
        )
        self.score_head = nn.Linear(16, 3)

    def attention(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = (self.key(x) * self.query).sum(-1) / math.sqrt(16.0)
        negative = torch.finfo(logits.dtype).min
        masked = torch.where(mask, logits, torch.full_like(logits, negative))
        row_max = masked.max(1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        mass = torch.exp(masked - row_max) * mask.to(logits.dtype)
        denominator = mass.sum(1, keepdim=True).clamp_min(1e-12)
        return (mass.unsqueeze(-1) * x).sum(1) / denominator

    def aggregate(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n = mask.sum(1).to(x.dtype)
        if self.name == "mean":
            core = masked_mean(x, mask)
        elif self.name == "sum":
            core = (x * mask.unsqueeze(-1)).sum(1)
        elif self.name == "mean_degree":
            standardized_degree = ((n - self.lambda0) / math.sqrt(self.lambda0)).unsqueeze(-1)
            core = torch.cat([masked_mean(x, mask), standardized_degree], dim=-1)
        elif self.name == "gat":
            core = self.attention(x, mask)
        else:  # pragma: no cover
            raise AssertionError(self.name)
        return torch.cat([core, empty_indicator(mask, x.dtype)], dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aggregate = self.aggregate(x, mask)
        hidden = self.trunk(aggregate)
        prediction = self.score_head(hidden)
        return prediction, aggregate, hidden


class NuisanceRegressor(nn.Module):
    def __init__(self, input_dim: int, location: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("input_location", location.detach().to(torch.float32))
        self.register_buffer("input_scale", scale.detach().to(torch.float32))
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.input_location) / self.input_scale)
