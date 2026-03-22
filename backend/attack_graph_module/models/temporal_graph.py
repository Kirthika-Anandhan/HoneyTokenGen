"""
Temporal graph components for the Attack Behaviour TGN.

Contains:
  - TemporalMemory      — GRU-based node memory that evolves across snapshots
  - TemporalAttention   — attention over past node states (optional upgrade)
  - SnapshotSequencer   — batches and pads snapshot sequences for training
  - TemporalEdgeEncoder — encodes time-delta between events as edge features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# 1. Temporal Memory (GRU per node type)
# ─────────────────────────────────────────────

class TemporalMemory(nn.Module):
    """
    Maintains a hidden state (memory) per node across temporal snapshots.

    How it works:
      - At each snapshot G(t), the GNN produces node embeddings h(t).
      - The GRU takes h(t) as input and the previous hidden state h(t-1)
        to produce a new hidden state that captures temporal context.
      - The updated state is projected back to hidden_dim and added
        residually to h(t) so the downstream heads see both current
        and historical context.

    Node counts change per snapshot (new IPs appear, old ones disappear).
    This class handles alignment automatically — new nodes start with
    zero memory, disappearing nodes are silently dropped.
    """

    def __init__(self, hidden_dim: int, mem_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mem_dim    = mem_dim

        self.node_types = ['ip', 'user', 'file', 'process']

        # One GRUCell per node type
        self.grus = nn.ModuleDict({
            ntype: nn.GRUCell(input_size=hidden_dim, hidden_size=mem_dim)
            for ntype in self.node_types
        })

        # Project memory back to hidden_dim for residual addition
        self.proj_back = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(mem_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            for ntype in self.node_types
        })

        # Gate: learned blend of current embedding vs memory projection
        self.gate = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            for ntype in self.node_types
        })

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        embeddings:   Dict[str, torch.Tensor],
        memory_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            embeddings:   GNN output {ntype: (N, hidden_dim)}
            memory_state: previous GRU hidden states {ntype: (M, mem_dim)}
                          M can differ from N — handled automatically.

        Returns:
            updated_embeddings: {ntype: (N, hidden_dim)}  — fused with memory
            new_memory_state:   {ntype: (N, mem_dim)}     — pass to next step
        """
        updated_embs  = {}
        new_mem_state = {}

        for ntype in self.node_types:
            emb = embeddings.get(ntype)
            if emb is None or emb.size(0) == 0:
                updated_embs[ntype]  = emb if emb is not None else torch.zeros(0, self.hidden_dim)
                new_mem_state[ntype] = torch.zeros(0, self.mem_dim, device=emb.device if emb is not None else 'cpu')
                continue

            N      = emb.size(0)
            device = emb.device

            # Align previous memory to current node count
            h_prev = self._align_memory(
                memory_state.get(ntype) if memory_state else None,
                N, device
            )

            # GRU update: h_new = GRU(emb, h_prev)
            h_new = self.grus[ntype](emb, h_prev)          # (N, mem_dim)
            new_mem_state[ntype] = h_new

            # Project memory back → hidden_dim
            mem_proj = self.proj_back[ntype](h_new)         # (N, hidden_dim)
            mem_proj = self.dropout(mem_proj)

            # Learned gate: how much to blend current embedding vs memory
            gate_input = torch.cat([emb, mem_proj], dim=-1)
            gate_val   = self.gate[ntype](gate_input)       # (N, hidden_dim)

            # Gated fusion: g * emb + (1-g) * mem_proj
            fused = gate_val * emb + (1.0 - gate_val) * mem_proj
            updated_embs[ntype] = fused

        return updated_embs, new_mem_state

    def _align_memory(
        self,
        prev_mem: Optional[torch.Tensor],
        n:        int,
        device:   torch.device,
    ) -> torch.Tensor:
        """
        Align previous memory tensor to current node count N.
        - If no previous memory: initialise with zeros.
        - If prev has fewer nodes than N: pad with zeros (new nodes).
        - If prev has more nodes than N: truncate (nodes left the window).
        """
        if prev_mem is None or prev_mem.size(0) == 0:
            return torch.zeros(n, self.mem_dim, device=device)

        M = prev_mem.size(0)
        prev_mem = prev_mem.to(device)

        if M == n:
            return prev_mem
        elif M > n:
            return prev_mem[:n]
        else:
            pad = torch.zeros(n - M, self.mem_dim, device=device)
            return torch.cat([prev_mem, pad], dim=0)

    def reset_memory(self) -> Dict[str, torch.Tensor]:
        """Return a fresh (zero) memory state."""
        return {ntype: torch.zeros(0, self.mem_dim) for ntype in self.node_types}


# ─────────────────────────────────────────────
# 2. Temporal Attention (optional upgrade over GRU)
# ─────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """
    Alternative to GRU memory: attends over the last K snapshot embeddings.

    Useful when you want the model to look back further than a single
    previous state — e.g., notice that an IP appeared 3 windows ago
    then disappeared, and is now back.

    Usage: replace TemporalMemory with this in AttackBehaviourTGN if
    you have GPU memory to spare and longer campaigns to learn from.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, history_len: int = 8):
        super().__init__()
        self.history_len = history_len
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=0.1, batch_first=True,
        )
        self.norm  = nn.LayerNorm(hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # History buffer: {ntype: list of tensors}
        self._history: Dict[str, List[torch.Tensor]] = {
            ntype: [] for ntype in ['ip', 'user', 'file', 'process']
        }

    def forward(
        self,
        embeddings: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], None]:
        """
        Attends over stored history and updates embeddings.
        Returns (updated_embeddings, None) — None for API compat with TemporalMemory.
        """
        updated = {}
        for ntype, emb in embeddings.items():
            if emb.size(0) == 0:
                updated[ntype] = emb
                continue

            history = self._history[ntype]

            if not history:
                # No history yet — pass through
                updated[ntype] = emb
            else:
                # Stack history: (T, N_max, D) — pad to common N
                max_n = max(h.size(0) for h in history)
                N     = emb.size(0)
                max_n = max(max_n, N)

                padded = []
                for h in history[-self.history_len:]:
                    if h.size(0) < max_n:
                        pad = torch.zeros(max_n - h.size(0), h.size(-1), device=emb.device)
                        h   = torch.cat([h, pad], dim=0)
                    padded.append(h[:max_n])

                # Current step as query, history as key/value
                # Reshape: treat each node independently
                # emb: (N, D) → query shape (N, 1, D)
                # history: (T, N, D) → (N, T, D) after transpose

                T       = len(padded)
                hist_t  = torch.stack(padded, dim=0)    # (T, max_n, D)
                q_emb   = emb[:N]                        # (N, D)

                # For each node, attend over its T historical representations
                hist_n  = hist_t[:, :N, :].permute(1, 0, 2)  # (N, T, D)
                q_n     = q_emb.unsqueeze(1)                  # (N, 1, D)

                attn_out, _ = self.attn(q_n, hist_n, hist_n)  # (N, 1, D)
                attn_out    = attn_out.squeeze(1)              # (N, D)

                h = self.norm(emb + attn_out)
                h = self.norm2(h + self.ff(h))
                updated[ntype] = h

            # Store current embedding in history
            self._history[ntype].append(emb.detach())
            if len(self._history[ntype]) > self.history_len:
                self._history[ntype].pop(0)

        return updated, None

    def reset_history(self):
        self._history = {ntype: [] for ntype in ['ip', 'user', 'file', 'process']}


# ─────────────────────────────────────────────
# 3. Temporal Edge Encoder
# ─────────────────────────────────────────────

class TemporalEdgeEncoder(nn.Module):
    """
    Encodes time-delta between events as additional edge features.

    Adds temporal context to edges: an edge that happens 5 seconds
    after a previous interaction is very different from one that happens
    3 hours later.

    Uses sinusoidal time encoding (like Transformer positional encoding)
    so the model can generalise to unseen time gaps.
    """

    def __init__(self, edge_feat_dim: int, time_dim: int = 16):
        super().__init__()
        self.time_dim = time_dim
        self.proj     = nn.Linear(edge_feat_dim + time_dim, edge_feat_dim)
        self.norm     = nn.LayerNorm(edge_feat_dim)

        # Learnable frequency bands for sinusoidal encoding
        self.freq = nn.Parameter(
            torch.randn(time_dim // 2) * 0.01
        )

    def _time_encode(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal encoding of time delta.
        delta_t: (E,) seconds since last event on this edge
        Returns: (E, time_dim)
        """
        delta_t  = delta_t.unsqueeze(-1).float()   # (E, 1)
        freqs    = self.freq.unsqueeze(0)            # (1, time_dim//2)
        angles   = delta_t * freqs                  # (E, time_dim//2)
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return encoding

    def forward(
        self,
        edge_attr: torch.Tensor,   # (E, edge_feat_dim)
        delta_t:   torch.Tensor,   # (E,) time since last event in seconds
    ) -> torch.Tensor:
        time_enc = self._time_encode(delta_t)            # (E, time_dim)
        combined = torch.cat([edge_attr, time_enc], dim=-1)
        return self.norm(self.proj(combined))


# ─────────────────────────────────────────────
# 4. Snapshot Sequencer (for training batching)
# ─────────────────────────────────────────────

class SnapshotSequencer:
    """
    Manages sequences of graph snapshots for training.

    Handles:
      - Padding sequences to the same length within a batch
      - Tracking which snapshots belong to which campaign
      - Generating negative samples for link prediction training
    """

    def __init__(self, max_seq_len: int = 20, device: str = 'cpu'):
        self.max_seq_len = max_seq_len
        self.device      = device

    def truncate_or_pad(
        self, snapshots: List, pad_value=None
    ) -> List:
        """Truncate long sequences, return as-is if shorter."""
        if len(snapshots) > self.max_seq_len:
            # Take the last max_seq_len snapshots (most recent context)
            return snapshots[-self.max_seq_len:]
        return snapshots

    def sample_negative_edges(
        self,
        pos_src:    torch.Tensor,   # (E,) positive source indices
        pos_dst:    torch.Tensor,   # (E,) positive destination indices
        n_nodes:    int,
        neg_ratio:  float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample negative edges for link prediction training.
        Corrupts destination nodes of positive edges randomly.

        Returns (neg_src, neg_dst) tensors.
        """
        n_neg   = int(len(pos_src) * neg_ratio)
        neg_src = pos_src[:n_neg]
        neg_dst = torch.randint(0, n_nodes, (n_neg,), device=pos_src.device)

        # Remove accidental positives
        pos_set  = set(zip(pos_src.tolist(), pos_dst.tolist()))
        keep     = [
            (s, d) for s, d in zip(neg_src.tolist(), neg_dst.tolist())
            if (s, d) not in pos_set
        ]
        if not keep:
            return neg_src, neg_dst

        neg_src_f = torch.tensor([x[0] for x in keep], dtype=torch.long, device=pos_src.device)
        neg_dst_f = torch.tensor([x[1] for x in keep], dtype=torch.long, device=pos_src.device)
        return neg_src_f, neg_dst_f

    def get_sequence_stats(self, snapshots: List) -> Dict:
        """Return summary statistics about a snapshot sequence."""
        if not snapshots:
            return {}

        node_counts = []
        edge_counts = []
        for snap in snapshots:
            total_nodes = sum(
                snap[nt].x.size(0) for nt in ['ip', 'user', 'file', 'process']
                if hasattr(snap[nt], 'x')
            )
            total_edges = sum(
                snap[src, rel, dst].edge_index.size(1)
                for (src, rel, dst) in snap.edge_types
            )
            node_counts.append(total_nodes)
            edge_counts.append(total_edges)

        return {
            'num_snapshots': len(snapshots),
            'avg_nodes':     sum(node_counts) / len(node_counts),
            'max_nodes':     max(node_counts),
            'avg_edges':     sum(edge_counts) / len(edge_counts),
            'max_edges':     max(edge_counts),
            'duration_sec':  getattr(snapshots[-1], 't_end', 0) - getattr(snapshots[0], 't_start', 0),
        }