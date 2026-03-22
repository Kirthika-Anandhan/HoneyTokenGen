"""
Output heads for the Attack Behaviour TGN.

Three heads:
  1. AnomalyHead       — per-node anomaly score in [0, 1]
  2. BehaviourHead     — graph-level attack stage classification
  3. LinkPredictor     — future edge existence prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


BEHAVIOUR_LABELS = {
    0: 'benign',
    1: 'reconnaissance',
    2: 'lateral_movement',
    3: 'privilege_escalation',
    4: 'exfiltration',
}


# ─────────────────────────────────────────────
# Head 1: Per-node anomaly scoring
# ─────────────────────────────────────────────

class AnomalyHead(nn.Module):
    """
    Per-node anomaly score in [0, 1].

    Each node type gets its own MLP so the model learns
    what "suspicious" means differently for an IP vs a user vs a file.

    Output: {ntype: (N, 1)} sigmoid scores
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.heads = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.SiLU(),
                nn.Linear(hidden_dim // 4, 1),
            )
            for ntype in ['ip', 'user', 'file', 'process']
        })

    def forward(
        self, x_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        out = {}
        for ntype, x in x_dict.items():
            if ntype in self.heads and x.size(0) > 0:
                out[ntype] = torch.sigmoid(self.heads[ntype](x))
        return out

    def get_top_suspicious_nodes(
        self,
        scores: Dict[str, torch.Tensor],
        top_k:  int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns the top-k most suspicious node indices per type.
        Useful for highlighting in the graph visualiser.
        """
        result = {}
        for ntype, s in scores.items():
            if s.size(0) > 0:
                k = min(top_k, s.size(0))
                topk_vals, topk_idx = torch.topk(s.squeeze(-1), k)
                result[ntype] = {
                    'indices': topk_idx,
                    'scores':  topk_vals,
                }
        return result


# ─────────────────────────────────────────────
# Head 2: Graph-level behaviour classification
# ─────────────────────────────────────────────

class BehaviourHead(nn.Module):
    """
    Graph-level behaviour classification.

    Strategy:
      - Mean-pool IP embeddings  (capture network-level activity)
      - Mean-pool User embeddings (capture account-level activity)
      - Concatenate → MLP → logits over 5 behaviour classes

    Classes: benign / recon / lateral / escalation / exfil

    Output: (1, num_classes) logits
    """

    def __init__(self, hidden_dim: int, num_classes: int = 5):
        super().__init__()
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, num_classes),
        )

    def forward(
        self, x_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        device = next(
            (v for v in x_dict.values() if v.size(0) > 0), torch.zeros(1)
        ).device

        hidden_dim = next(
            (v.size(-1) for v in x_dict.values() if v.size(0) > 0), 128
        )

        ip_pool = (
            x_dict['ip'].mean(0)
            if 'ip' in x_dict and x_dict['ip'].size(0) > 0
            else torch.zeros(hidden_dim, device=device)
        )
        user_pool = (
            x_dict['user'].mean(0)
            if 'user' in x_dict and x_dict['user'].size(0) > 0
            else torch.zeros(hidden_dim, device=device)
        )

        pooled  = torch.cat([ip_pool, user_pool], dim=-1)
        h       = self.pool_proj(pooled)
        logits  = self.classifier(h)
        return logits.unsqueeze(0)   # (1, num_classes)

    def predict(
        self, x_dict: Dict[str, torch.Tensor]
    ) -> Dict:
        """Convenience wrapper — returns label + probabilities."""
        with torch.no_grad():
            logits = self.forward(x_dict)
            probs  = F.softmax(logits, dim=-1).squeeze(0)
            pred   = probs.argmax().item()
        return {
            'label':      BEHAVIOUR_LABELS[pred],
            'confidence': probs[pred].item(),
            'distribution': {
                BEHAVIOUR_LABELS[i]: probs[i].item()
                for i in range(len(BEHAVIOUR_LABELS))
            },
        }


# ─────────────────────────────────────────────
# Head 3: Link prediction (future edge)
# ─────────────────────────────────────────────

class LinkPredictor(nn.Module):
    """
    Predict whether a future edge (src, dst) will form.

    Used to forecast:
      - Which IPs will be targeted next (lateral movement)
      - Which files will be accessed (exfiltration targets)
      - New C2 connections forming

    Takes source and destination node embeddings,
    concatenates them, runs through MLP → sigmoid probability.

    Output: (E,) probabilities per candidate edge
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        src_emb: torch.Tensor,   # (E, hidden_dim)
        dst_emb: torch.Tensor,   # (E, hidden_dim)
    ) -> torch.Tensor:            # (E, 1)
        h = torch.cat([src_emb, dst_emb], dim=-1)
        return torch.sigmoid(self.mlp(h))

    def predict_top_k(
        self,
        node_embs:  torch.Tensor,   # (N, hidden_dim) — same type src & dst
        k:          int = 10,
        mask_self:  bool = True,
    ) -> torch.Tensor:
        """
        Score all N×N candidate edges and return top-k (src, dst, score).
        Useful for forecasting next lateral-move targets.
        """
        N = node_embs.size(0)
        src_idx = torch.arange(N).repeat_interleave(N)
        dst_idx = torch.arange(N).repeat(N)

        src_emb = node_embs[src_idx]
        dst_emb = node_embs[dst_idx]

        scores = self.forward(src_emb, dst_emb).squeeze(-1)

        if mask_self:
            self_mask = (src_idx == dst_idx)
            scores[self_mask] = -1.0

        topk_scores, topk_flat = torch.topk(scores, k=min(k, scores.size(0)))
        topk_src = src_idx[topk_flat]
        topk_dst = dst_idx[topk_flat]

        return torch.stack([topk_src, topk_dst, topk_scores], dim=-1)