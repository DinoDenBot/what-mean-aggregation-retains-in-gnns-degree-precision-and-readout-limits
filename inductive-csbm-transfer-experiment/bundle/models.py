from __future__ import annotations

import torch
from torch import nn


def _segment(values: torch.Tensor, degree: torch.Tensor, reduction: str) -> torch.Tensor:
    result = torch.segment_reduce(values, reduction, lengths=degree)
    return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def neighbor_statistics(
    x: torch.Tensor,
    edge_src: torch.Tensor,
    degree: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    messages = x[edge_src]
    summed = _segment(messages, degree, "sum")
    mean = _segment(messages, degree, "mean")
    maximum = _segment(messages, degree, "max")
    minimum = _segment(messages, degree, "min")
    mean_square = _segment(messages.square(), degree, "mean")
    standard_deviation = torch.sqrt((mean_square - mean.square()).clamp_min(0.0) + 1e-12)
    nonempty = (degree > 0).unsqueeze(-1)
    standard_deviation = torch.where(nonempty, standard_deviation, torch.zeros_like(standard_deviation))
    return summed, mean, standard_deviation, maximum, minimum


class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        architecture: str,
        input_dimension: int,
        output_dimension: int,
        degree_location: float,
        degree_scale: float,
        pna_delta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if architecture not in {"mean", "mean_degree", "sum", "pna"}:
            raise ValueError(f"unknown architecture: {architecture}")
        self.architecture = architecture
        self.register_buffer("degree_location", torch.tensor(float(degree_location), dtype=torch.float32))
        self.register_buffer("degree_scale", torch.tensor(float(degree_scale), dtype=torch.float32))
        self.register_buffer("pna_delta", torch.tensor(float(pna_delta), dtype=torch.float32))
        aggregate_multiplier = 12 if architecture == "pna" else 1
        aggregate_extra = 1 if architecture == "mean_degree" else 0
        affine_input = input_dimension + aggregate_multiplier * input_dimension + aggregate_extra
        self.affine = nn.Linear(affine_input, output_dimension)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def aggregate(self, x: torch.Tensor, edge_src: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        summed, mean, standard_deviation, maximum, minimum = neighbor_statistics(x, edge_src, degree)
        if self.architecture == "sum":
            return summed
        if self.architecture == "mean":
            return mean
        log_degree = torch.log1p(degree.to(x.dtype))
        if self.architecture == "mean_degree":
            standardized = (log_degree - self.degree_location.to(x.dtype)) / self.degree_scale.to(x.dtype)
            return torch.cat([mean, standardized.unsqueeze(-1)], dim=-1)
        statistics = torch.cat([mean, standard_deviation, maximum, minimum], dim=-1)
        delta = self.pna_delta.to(x.dtype).clamp_min(1e-8)
        amplification = log_degree / delta
        attenuation = torch.where(log_degree > 0, delta / log_degree, torch.zeros_like(log_degree))
        return torch.cat([
            statistics,
            statistics * amplification.unsqueeze(-1),
            statistics * attenuation.unsqueeze(-1),
        ], dim=-1)

    def forward(self, x: torch.Tensor, edge_src: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        aggregate = self.aggregate(x, edge_src, degree)
        return self.dropout(self.activation(self.affine(torch.cat([x, aggregate], dim=-1))))


class InductiveGNN(nn.Module):
    """Two-layer classifier whose forward interface intentionally excludes labels."""

    def __init__(
        self,
        architecture: str,
        input_dimension: int,
        hidden_width: int,
        degree_location: float,
        degree_scale: float,
        pna_delta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.layer1 = MessagePassingLayer(
            architecture, input_dimension, hidden_width,
            degree_location, degree_scale, pna_delta, dropout,
        )
        self.layer2 = MessagePassingLayer(
            architecture, hidden_width, hidden_width,
            degree_location, degree_scale, pna_delta, dropout,
        )
        self.classifier = nn.Linear(hidden_width, 1)

    def forward(self, x: torch.Tensor, edge_src: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        hidden = self.layer1(x, edge_src, degree)
        hidden = self.layer2(hidden, edge_src, degree)
        return self.classifier(hidden).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

