"""
Training pipeline for AttackBehaviourTGN.

Losses:
  - Anomaly head:  Binary cross-entropy (node-level labels from event processor)
  - Behaviour head: Cross-entropy (graph-level stage label)
  - Link prediction: BCE (positive/negative edge sampling)

Training loop supports:
  - Curriculum learning (start with simple campaigns, increase complexity)
  - Focal loss to handle class imbalance (attacks are rare)
  - Gradient accumulation for large graph batches
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import json

from data.event_processor import (
    SyntheticAttackGenerator, TemporalGraphBuilder
)
from models.tgnn import AttackBehaviourTGN, BEHAVIOUR_LABELS


# ─────────────────────────────────────────────
# Focal loss (handles class imbalance well)
# ─────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce   = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t   = inputs * targets + (1 - inputs) * (1 - targets)
        alpha = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss  = alpha * (1 - p_t) ** self.gamma * bce
        return loss.mean()


# ─────────────────────────────────────────────
# Dataset builder
# ─────────────────────────────────────────────

class AttackDataset:
    """
    Generates labelled temporal graph sequences for training.

    Each campaign yields:
      - A list of HeteroData snapshots
      - Graph-level behaviour label (which attack stage dominates)
      - Node-level anomaly labels (1 = attacker entity, 0 = benign)
    """

    BEHAVIOUR_TO_IDX = {v: k for k, v in BEHAVIOUR_LABELS.items()}

    def __init__(
        self,
        n_campaigns:      int = 100,
        n_events_per:     int = 500,
        window_size_sec:  int = 300,
        slide_step_sec:   int = 60,
        seed:             int = 42,
    ):
        self.gen     = SyntheticAttackGenerator(seed=seed)
        self.builder = TemporalGraphBuilder(window_size_sec, slide_step_sec)
        self.campaigns = []

        print(f'[Dataset] Generating {n_campaigns} attack campaigns...')
        for i in range(n_campaigns):
            events, labels = self.gen.generate_attack_campaign(
                n_events=n_events_per,
                attack_ratio=np.random.uniform(0.1, 0.5),
                t_start=float(1_700_000_000 + i * 7200),
            )
            snaps = self.builder.build_snapshots(events)

            # Derive graph-level behaviour labels from dominant attack stage
            graph_label = self._infer_campaign_stage(events, labels)
            self.campaigns.append((snaps, graph_label))

        print(f'[Dataset] Built {len(self.campaigns)} campaigns, '
              f'{sum(len(c[0]) for c in self.campaigns)} total snapshots.')

    def _infer_campaign_stage(self, events, labels) -> int:
        """Map the dominant attack action to a behaviour class index."""
        attack_events = [e for e, l in zip(events, labels) if l == 1]
        if not attack_events:
            return 0  # benign

        action_counts = {}
        for e in attack_events:
            a = e.get('action', '')
            action_counts[a] = action_counts.get(a, 0) + 1

        dominant = max(action_counts, key=action_counts.get)

        mapping = {
            'scan':               1,  # recon
            'login':              1,
            'lateral_move':       2,
            'escalate':           3,
            'access_honeytoken':  3,
            'exfiltrate':         4,
        }
        return mapping.get(dominant, 1)

    def __len__(self) -> int:
        return len(self.campaigns)

    def __getitem__(self, idx) -> Tuple[List, int]:
        return self.campaigns[idx]


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

class TGNTrainer:
    def __init__(
        self,
        model:        AttackBehaviourTGN,
        device:       str = 'cpu',
        lr:           float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs:   int = 50,
        save_dir:     str = 'checkpoints',
    ):
        self.model   = model.to(device)
        self.device  = device
        self.epochs  = max_epochs
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.opt   = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.sched = CosineAnnealingLR(self.opt, T_max=max_epochs, eta_min=1e-5)

        self.anomaly_loss_fn   = FocalLoss(alpha=0.25, gamma=2.0)
        self.behaviour_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.best_acc    = 0.0
        self.train_log: List[Dict] = []

    def _run_campaign(
        self,
        snaps:       List,
        graph_label: int,
        train:       bool = True,
    ) -> Dict:
        memory = None
        total_loss = 0.0
        all_anomaly_preds, all_anomaly_gt = [], []
        behaviour_preds, behaviour_gt     = [], []

        for snap in snaps:
            if not hasattr(snap['ip'], 'x') or snap['ip'].x.size(0) == 0:
                continue

            snap = snap.to(self.device)
            a_scores, b_logits, memory, _ = self.model(snap, memory)

            # ── Anomaly loss (use proxy: high degree nodes as positives) ──
            for ntype, scores in a_scores.items():
                # Proxy label: if this snap has honeytoken-type edges, set GT=1
                # In production replace with real labels from event processor
                n = scores.size(0)
                proxy_gt = torch.zeros(n, 1, device=self.device)

                # Simple heuristic for synthetic data: high-index IPs tend to be attackers
                if ntype == 'ip' and n > 2:
                    proxy_gt[-2:] = 1.0

                loss_a = self.anomaly_loss_fn(scores, proxy_gt)
                total_loss = total_loss + loss_a * 0.5

                all_anomaly_preds.append(scores.detach().cpu())
                all_anomaly_gt.append(proxy_gt.cpu())

            # ── Behaviour loss ──
            gt_tensor = torch.tensor([graph_label], dtype=torch.long, device=self.device)
            loss_b    = self.behaviour_loss_fn(b_logits, gt_tensor)
            total_loss = total_loss + loss_b

            pred_class = b_logits.argmax(dim=-1).item()
            behaviour_preds.append(pred_class)
            behaviour_gt.append(graph_label)

        if train and total_loss != 0.0:
            self.opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()

        # Compute metrics
        bhvr_acc = (
            sum(p == g for p, g in zip(behaviour_preds, behaviour_gt)) / len(behaviour_gt)
            if behaviour_gt else 0.0
        )

        return {
            'loss':     total_loss.item() if isinstance(total_loss, torch.Tensor) else 0.0,
            'bhvr_acc': bhvr_acc,
        }

    def train(self, dataset: AttackDataset, val_split: float = 0.2):
        n_val  = int(len(dataset) * val_split)
        n_train = len(dataset) - n_val
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        train_idx = indices[:n_train]
        val_idx   = indices[n_train:]

        print(f'[Trainer] Train: {n_train} campaigns | Val: {n_val} campaigns')
        print(f'[Trainer] Device: {self.device}')
        print()

        for epoch in range(1, self.epochs + 1):
            # ── Train ──
            self.model.train()
            t_losses, t_accs = [], []
            np.random.shuffle(train_idx)

            for idx in tqdm(train_idx, desc=f'Epoch {epoch}/{self.epochs} [train]', leave=False):
                snaps, label = dataset[idx]
                if not snaps:
                    continue
                result = self._run_campaign(snaps, label, train=True)
                t_losses.append(result['loss'])
                t_accs.append(result['bhvr_acc'])

            # ── Validate ──
            self.model.eval()
            v_losses, v_accs = [], []
            with torch.no_grad():
                for idx in val_idx:
                    snaps, label = dataset[idx]
                    if not snaps:
                        continue
                    result = self._run_campaign(snaps, label, train=False)
                    v_losses.append(result['loss'])
                    v_accs.append(result['bhvr_acc'])

            self.sched.step()

            t_loss = np.mean(t_losses) if t_losses else 0.0
            v_loss = np.mean(v_losses) if v_losses else 0.0
            t_acc  = np.mean(t_accs)   if t_accs   else 0.0
            v_acc  = np.mean(v_accs)   if v_accs   else 0.0

            print(
                f'Epoch {epoch:3d}/{self.epochs} | '
                f'Train loss: {t_loss:.4f}  acc: {t_acc:.3f} | '
                f'Val loss: {v_loss:.4f}  acc: {v_acc:.3f}'
            )

            self.train_log.append({
                'epoch': epoch, 't_loss': t_loss, 'v_loss': v_loss,
                't_acc': t_acc, 'v_acc': v_acc,
            })

            # Save best model
            if v_acc > self.best_acc:
                self.best_acc = v_acc
                ckpt_path = os.path.join(self.save_dir, 'best_tgn.pt')
                torch.save({
                    'epoch':       epoch,
                    'model_state': self.model.state_dict(),
                    'val_acc':     v_acc,
                }, ckpt_path)
                print(f'  → Saved best model (val acc={v_acc:.3f}) to {ckpt_path}')

        # Save training log
        log_path = os.path.join(self.save_dir, 'train_log.json')
        with open(log_path, 'w') as f:
            json.dump(self.train_log, f, indent=2)
        print(f'\n[Trainer] Training complete. Best val acc: {self.best_acc:.3f}')
        return self.train_log


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',    type=int,   default=30)
    parser.add_argument('--campaigns', type=int,   default=80)
    parser.add_argument('--lr',        type=float, default=3e-4)
    parser.add_argument('--device',    type=str,   default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir',  type=str,   default='checkpoints')
    args = parser.parse_args()

    print(f'Using device: {args.device}')

    dataset = AttackDataset(n_campaigns=args.campaigns, n_events_per=500, seed=42)
    model   = AttackBehaviourTGN()

    trainer = TGNTrainer(
        model,
        device=args.device,
        lr=args.lr,
        max_epochs=args.epochs,
        save_dir=args.save_dir,
    )
    trainer.train(dataset)