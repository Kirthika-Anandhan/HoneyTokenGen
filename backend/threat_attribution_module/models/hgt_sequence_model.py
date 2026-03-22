"""
Heterogeneous Graph Transformer + temporal Transformer for attribution & profiling.

Stacks HGTConv layers on each snapshot, pools to a sequence embedding, then
TransformerEncoder for graph-sequence modeling. Multi-task heads:
  - Campaign-level threat actor attribution
  - Per-snapshot attack stage (profile)
  - Next-stage prediction from sequence context
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

# Resolve attack_graph_module (same layout as attack_graph_router)
_AG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "attack_graph_module"))
if _AG not in sys.path:
    sys.path.insert(0, _AG)

from models.tgnn import NodeProjections  # noqa: E402
from data.event_processor import (  # noqa: E402
    IP_FEAT_DIM,
    USER_FEAT_DIM,
    FILE_FEAT_DIM,
    PROCESS_FEAT_DIM,
)

NODE_TYPES = ["ip", "user", "file", "process"]
_RAW_DIM = {
    "ip": IP_FEAT_DIM,
    "user": USER_FEAT_DIM,
    "file": FILE_FEAT_DIM,
    "process": PROCESS_FEAT_DIM,
}
EDGE_TYPES = [
    ("ip", "connects_to", "ip"),
    ("ip", "scans", "ip"),
    ("user", "accesses", "file"),
    ("user", "runs", "process"),
    ("ip", "authenticates", "user"),
    ("process", "opens", "file"),
    ("user", "lateral_moves", "ip"),
]
METADATA = (NODE_TYPES, EDGE_TYPES)

ATTRIBUTION_LABELS = {
    0: "undetermined",
    1: "external_network_attacker",
    2: "compromised_account",
    3: "privileged_insider",
    4: "automated_tooling",
}

PROFILE_LABELS = {
    0: "benign",
    1: "reconnaissance",
    2: "lateral_movement",
    3: "privilege_escalation",
    4: "exfiltration",
}


def _edge_dict(data: HeteroData) -> Dict:
    out = {}
    for et in EDGE_TYPES:
        if et in data.edge_types:
            ei = data[et].edge_index
            out[et] = ei if ei.numel() > 0 else torch.zeros(2, 0, dtype=torch.long, device=ei.device)
    return out


def _x_dict(data: HeteroData, device: torch.device) -> Dict[str, torch.Tensor]:
    xd = {}
    for nt in NODE_TYPES:
        if hasattr(data[nt], "x") and data[nt].x is not None:
            xd[nt] = data[nt].x.to(device)
        else:
            xd[nt] = torch.zeros(0, _RAW_DIM[nt], device=device)
    return xd


def pool_hetero(
    h_dict: Dict[str, torch.Tensor], hidden_dim: int, device: torch.device
) -> torch.Tensor:
    parts = []
    for nt in NODE_TYPES:
        if nt in h_dict and h_dict[nt].size(0) > 0:
            parts.append(h_dict[nt].mean(dim=0))
        else:
            parts.append(torch.zeros(hidden_dim, device=device))
    return torch.cat(parts, dim=-1)


class ThreatAttributionSequenceModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        num_hgt_layers: int = 2,
        num_heads: int = 4,
        tf_layers: int = 2,
        tf_heads: int = 4,
        dropout: float = 0.2,
        num_attribution: int = 5,
        num_profile: int = 5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj = NodeProjections(hidden_dim)

        self.hgt_layers = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=METADATA,
                    heads=num_heads,
                )
                for _ in range(num_hgt_layers)
            ]
        )
        self.hgt_norms = nn.ModuleDict(
            {nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES}
        )

        self.snapshot_in = nn.Linear(hidden_dim * 4, hidden_dim)
        self.snapshot_act = nn.SiLU()

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=tf_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.sequence_encoder = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)
        self.pos_embed = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)

        self.head_attribution = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_attribution),
        )
        self.head_profile = nn.Linear(hidden_dim, num_profile)
        self.head_next = nn.Linear(hidden_dim, num_profile)

    def encode_snapshot(self, data: HeteroData, device: torch.device) -> torch.Tensor:
        data = data.to(device)
        x_dict = _x_dict(data, device)
        edge_index_dict = _edge_dict(data)

        h = self.proj(x_dict)
        for nt in NODE_TYPES:
            if nt in h and h[nt].size(0) == 0:
                h[nt] = torch.zeros(0, self.hidden_dim, device=device)
            elif nt not in h:
                h[nt] = torch.zeros(0, self.hidden_dim, device=device)

        for conv in self.hgt_layers:
            h = conv(h, edge_index_dict)
            for nt in NODE_TYPES:
                if nt in h and h[nt].size(0) > 0:
                    h[nt] = F.silu(h[nt])
                    h[nt] = self.hgt_norms[nt](h[nt])

        pooled = pool_hetero(h, self.hidden_dim, device)
        z = self.snapshot_act(self.snapshot_in(pooled))
        return z

    def forward(
        self, snapshots: List[HeteroData], device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            attr_logits: (1, num_attr) campaign-level
            profile_logits: (T, num_profile) per snapshot
            next_logits: (T, num_profile) predict next stage from each position
        """
        if not snapshots:
            raise ValueError("snapshots list is empty")
        device = device or next(self.parameters()).device
        embs = [self.encode_snapshot(s, device) for s in snapshots]
        T = len(embs)
        x = torch.stack(embs, dim=0).unsqueeze(0)  # (1, T, H)
        if T <= self.pos_embed.size(1):
            x = x + self.pos_embed[:, :T, :]
        else:
            x = x + self.pos_embed[:, :1, :].expand(1, T, -1)

        h = self.sequence_encoder(x)  # (1, T, H)
        seq = h.squeeze(0)  # (T, H)

        global_vec = seq.mean(dim=0, keepdim=True)  # (1, H)
        attr_logits = self.head_attribution(global_vec)

        profile_logits = self.head_profile(seq)
        next_logits = self.head_next(seq)
        return attr_logits, profile_logits, next_logits
