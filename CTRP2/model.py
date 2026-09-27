# -*- coding: utf-8 -*-
"""Four-view shared_consensus_common_complementary model.

Four independent GCN encoders feed a shared common projector and view-specific
complementary projectors. A parameter-shared GCN reference branch aligns common
representations at the same atom in each view via cosine alignment loss.
The shared reference branch is auxiliary and does not enter prediction.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class RNAEncoder(nn.Module):
    def __init__(self, rna_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rna_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        if rna.dim() == 3:
            rna = rna.squeeze(1)
        return self.net(rna.float())


class GCNEncoder(nn.Module):
    """Two-layer GCN; the second layer has no activation or dropout."""

    def __init__(self, atom_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(atom_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = self.dropout(F.relu(self.norm1(h)))
        return self.conv2(h, edge_index)


class ProjectionMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResponseHead(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        drug_vector: torch.Tensor,
        cell_vector: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(torch.cat([drug_vector, cell_vector], dim=-1)).view(-1)


def alignment_loss(
    shared_nodes: Tuple[torch.Tensor, ...],
    common_nodes: Tuple[torch.Tensor, ...],
) -> torch.Tensor:
    losses = [
        (1.0 - F.cosine_similarity(shared, common, dim=-1, eps=1e-8)).mean()
        for shared, common in zip(shared_nodes, common_nodes)
    ]
    return torch.stack(losses).mean()


class CommonComplementary4ViewEncoder(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even.")

        component_dim = hidden_dim // 2

        self.independent_encoders = nn.ModuleList(
            [GCNEncoder(atom_dim, hidden_dim, dropout) for _ in range(4)]
        )
        self.common_projector = ProjectionMLP(
            hidden_dim, component_dim, dropout
        )
        self.complementary_projectors = nn.ModuleList(
            [
                ProjectionMLP(hidden_dim, component_dim, dropout)
                for _ in range(4)
            ]
        )

        self.shared_encoder = GCNEncoder(atom_dim, hidden_dim, dropout)
        self.shared_projector = ProjectionMLP(
            hidden_dim, component_dim, dropout
        )

        self.fuse = FusionMLP(hidden_dim * 4, hidden_dim, dropout)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        common_nodes = []
        shared_nodes = []
        view_vectors = []

        for scale in range(4):
            edge_index = getattr(data, f"edge_index_s{scale}")
            full_nodes = self.independent_encoders[scale](data.x, edge_index)

            common = self.common_projector(full_nodes)
            complementary = self.complementary_projectors[scale](full_nodes)
            common_nodes.append(common)

            view_nodes = torch.cat([common, complementary], dim=-1)
            view_vectors.append(global_mean_pool(view_nodes, data.batch))

            shared_full = self.shared_encoder(data.x, edge_index)
            shared_nodes.append(self.shared_projector(shared_full))

        drug_vector = self.fuse(torch.cat(view_vectors, dim=-1))
        align = alignment_loss(tuple(shared_nodes), tuple(common_nodes))

        return drug_vector, align


class CellDrugRegressor(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        rna_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.drug_encoder = CommonComplementary4ViewEncoder(
            atom_dim=atom_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.rna_encoder = RNAEncoder(rna_dim, hidden_dim, dropout)
        self.response_head = ResponseHead(hidden_dim, dropout)

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        drug_vector, align = self.drug_encoder(data)
        cell_vector = self.rna_encoder(data.rna)
        prediction = self.response_head(drug_vector, cell_vector)
        return prediction, align
