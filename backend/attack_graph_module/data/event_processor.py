"""
Event Processor — converts raw honeytoken/network logs into temporal graph snapshots.
Each snapshot G(t) is a PyG HeteroData object with typed nodes and edges.
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch_geometric.data import HeteroData
from typing import List, Dict, Tuple, Optional
import hashlib
import json
from datetime import datetime


# ──────────────────────────────────────────────
# Node & Edge feature dims
# ──────────────────────────────────────────────
IP_FEAT_DIM       = 16   # IP address features
USER_FEAT_DIM     = 12   # user account features
FILE_FEAT_DIM     = 10   # file/resource features
PROCESS_FEAT_DIM  = 14   # process features
EDGE_FEAT_DIM     = 20   # interaction features


class EntityRegistry:
    """Maps string entity IDs → integer indices (per snapshot or global)."""

    def __init__(self):
        self.entities: Dict[str, Dict[str, int]] = defaultdict(dict)

    def get_or_add(self, entity_type: str, entity_id: str) -> int:
        if entity_id not in self.entities[entity_type]:
            self.entities[entity_type][entity_id] = len(self.entities[entity_type])
        return self.entities[entity_type][entity_id]

    def size(self, entity_type: str) -> int:
        return len(self.entities[entity_type])

    def reset(self):
        self.entities = defaultdict(dict)


def _ip_to_features(ip: str) -> torch.Tensor:
    """Encode IP address into a fixed-length feature vector."""
    feats = torch.zeros(IP_FEAT_DIM)
    try:
        parts = [int(x) for x in ip.split('.')]
        for i, p in enumerate(parts[:4]):
            feats[i] = p / 255.0
        # Private range flags
        feats[4] = 1.0 if parts[0] == 10 else 0.0
        feats[5] = 1.0 if (parts[0] == 172 and 16 <= parts[1] <= 31) else 0.0
        feats[6] = 1.0 if (parts[0] == 192 and parts[1] == 168) else 0.0
        feats[7] = 1.0 if parts[0] == 127 else 0.0  # loopback
    except Exception:
        pass
    # Hash-based features for the remaining slots
    h = int(hashlib.md5(ip.encode()).hexdigest(), 16)
    for i in range(8, IP_FEAT_DIM):
        feats[i] = ((h >> i) & 0xFF) / 255.0
    return feats


def _user_to_features(user: str, is_admin: bool = False, failed_logins: int = 0) -> torch.Tensor:
    feats = torch.zeros(USER_FEAT_DIM)
    h = int(hashlib.md5(user.encode()).hexdigest(), 16)
    for i in range(8):
        feats[i] = ((h >> i) & 0xFF) / 255.0
    feats[8]  = 1.0 if is_admin else 0.0
    feats[9]  = min(failed_logins / 10.0, 1.0)
    feats[10] = 1.0 if 'service' in user.lower() else 0.0
    feats[11] = 1.0 if 'admin' in user.lower() or 'root' in user.lower() else 0.0
    return feats


def _file_to_features(path: str, sensitivity: float = 0.5) -> torch.Tensor:
    feats = torch.zeros(FILE_FEAT_DIM)
    h = int(hashlib.md5(path.encode()).hexdigest(), 16)
    for i in range(6):
        feats[i] = ((h >> i) & 0xFF) / 255.0
    feats[6] = sensitivity
    feats[7] = 1.0 if any(x in path for x in ['/etc', '/root', 'password', 'shadow', 'key']) else 0.0
    feats[8] = 1.0 if any(x in path for x in ['.sh', '.exe', '.py', '.bat']) else 0.0
    feats[9] = 1.0 if 'honeytoken' in path.lower() or 'honey' in path.lower() else 0.0
    return feats


def _process_to_features(proc_name: str, pid: int = 0, ppid: int = 0) -> torch.Tensor:
    feats = torch.zeros(PROCESS_FEAT_DIM)
    h = int(hashlib.md5(proc_name.encode()).hexdigest(), 16)
    for i in range(8):
        feats[i] = ((h >> i) & 0xFF) / 255.0
    feats[8]  = min(pid / 65535.0, 1.0)
    feats[9]  = min(ppid / 65535.0, 1.0)
    feats[10] = 1.0 if proc_name in ['cmd.exe', 'powershell', 'bash', 'sh'] else 0.0
    feats[11] = 1.0 if proc_name in ['nmap', 'masscan', 'hydra', 'metasploit'] else 0.0
    feats[12] = 1.0 if 'python' in proc_name or 'perl' in proc_name else 0.0
    feats[13] = 1.0 if 'curl' in proc_name or 'wget' in proc_name else 0.0
    return feats


def _edge_features(
    action: str,
    port: int = 0,
    bytes_transferred: int = 0,
    duration_ms: float = 0.0,
    success: bool = True,
    count: int = 1,
) -> torch.Tensor:
    ACTION_MAP = {
        'login': 0, 'logout': 1, 'read': 2, 'write': 3, 'execute': 4,
        'connect': 5, 'scan': 6, 'download': 7, 'upload': 8,
        'access_honeytoken': 9, 'lateral_move': 10, 'escalate': 11,
        'exfiltrate': 12, 'c2_beacon': 13, 'brute_force': 14,
    }
    feats = torch.zeros(EDGE_FEAT_DIM)
    action_idx = ACTION_MAP.get(action, 15)
    # One-hot encode action type (first 16 dims)
    if action_idx < EDGE_FEAT_DIM:
        feats[action_idx] = 1.0
    feats[15] = min(port / 65535.0, 1.0)
    feats[16] = min(np.log1p(bytes_transferred) / 20.0, 1.0)
    feats[17] = min(duration_ms / 10000.0, 1.0)
    feats[18] = 1.0 if success else 0.0
    feats[19] = min(count / 100.0, 1.0)
    return feats


# ──────────────────────────────────────────────
# Core: Raw events → Graph snapshots
# ──────────────────────────────────────────────

class TemporalGraphBuilder:
    """
    Converts a stream of raw security events into a sequence of PyG HeteroData
    temporal graph snapshots using a sliding window.

    Args:
        window_size_sec: duration of each snapshot window in seconds
        slide_step_sec:  how far to slide the window for the next snapshot
    """

    NODE_TYPES  = ['ip', 'user', 'file', 'process']
    EDGE_TYPES  = [
        ('ip',      'connects_to',     'ip'),
        ('user',    'accesses',        'file'),
        ('user',    'runs',            'process'),
        ('ip',      'authenticates',   'user'),
        ('process', 'opens',           'file'),
        ('user',    'lateral_moves',   'ip'),
        ('ip',      'scans',           'ip'),
    ]

    def __init__(
        self,
        window_size_sec: int = 300,   # 5 min default
        slide_step_sec:  int = 60,    # 1 min slide
    ):
        self.window_size = window_size_sec
        self.slide_step  = slide_step_sec

    def build_snapshots(self, events: List[Dict]) -> List[HeteroData]:
        """
        Args:
            events: list of dicts with keys:
                    timestamp (float/unix), src_ip, dst_ip, user, file_path,
                    process, action, port, bytes, duration_ms, success
        Returns:
            List of HeteroData objects (one per time window)
        """
        if not events:
            return []

        # Sort by timestamp
        events = sorted(events, key=lambda e: e['timestamp'])
        t_min = events[0]['timestamp']
        t_max = events[-1]['timestamp']

        snapshots = []
        t_start = t_min

        while t_start < t_max:
            t_end = t_start + self.window_size
            window_events = [e for e in events if t_start <= e['timestamp'] < t_end]

            if window_events:
                snap = self._build_single_snapshot(window_events, t_start, t_end)
                if snap is not None:
                    snapshots.append(snap)

            t_start += self.slide_step

        return snapshots

    def _build_single_snapshot(
        self,
        events: List[Dict],
        t_start: float,
        t_end: float,
    ) -> Optional[HeteroData]:
        """Build a single HeteroData graph from events in one time window."""
        registry = EntityRegistry()
        data = HeteroData()

        # ── Collect node features ──
        ip_feats:      Dict[int, torch.Tensor] = {}
        user_feats:    Dict[int, torch.Tensor] = {}
        file_feats:    Dict[int, torch.Tensor] = {}
        process_feats: Dict[int, torch.Tensor] = {}

        # ── Collect edge indices + features per relation ──
        edges: Dict[Tuple, Tuple[List[int], List[int], List[torch.Tensor]]] = {
            rel: ([], [], []) for rel in self.EDGE_TYPES
        }

        for ev in events:
            src_ip  = ev.get('src_ip', '')
            dst_ip  = ev.get('dst_ip', '')
            user    = ev.get('user', '')
            fpath   = ev.get('file_path', '')
            proc    = ev.get('process', '')
            action  = ev.get('action', 'connect')
            port    = int(ev.get('port', 0))
            nbytes  = int(ev.get('bytes', 0))
            dur     = float(ev.get('duration_ms', 0.0))
            success = bool(ev.get('success', True))

            edge_f  = _edge_features(action, port, nbytes, dur, success)

            # Register nodes & collect features
            if src_ip:
                idx = registry.get_or_add('ip', src_ip)
                ip_feats[idx] = _ip_to_features(src_ip)
            if dst_ip:
                idx = registry.get_or_add('ip', dst_ip)
                ip_feats[idx] = _ip_to_features(dst_ip)
            if user:
                idx = registry.get_or_add('user', user)
                user_feats[idx] = _user_to_features(user)
            if fpath:
                idx = registry.get_or_add('file', fpath)
                file_feats[idx] = _file_to_features(fpath)
            if proc:
                idx = registry.get_or_add('process', proc)
                process_feats[idx] = _process_to_features(proc)

            # Build edges per action type
            if src_ip and dst_ip:
                src_i = registry.get_or_add('ip', src_ip)
                dst_i = registry.get_or_add('ip', dst_ip)
                if action == 'scan':
                    e_src, e_dst, e_f = edges[('ip', 'scans', 'ip')]
                else:
                    e_src, e_dst, e_f = edges[('ip', 'connects_to', 'ip')]
                e_src.append(src_i); e_dst.append(dst_i); e_f.append(edge_f)

            if user and fpath:
                src_i = registry.get_or_add('user', user)
                dst_i = registry.get_or_add('file', fpath)
                e_src, e_dst, e_f = edges[('user', 'accesses', 'file')]
                e_src.append(src_i); e_dst.append(dst_i); e_f.append(edge_f)

            if user and proc:
                src_i = registry.get_or_add('user', user)
                dst_i = registry.get_or_add('process', proc)
                e_src, e_dst, e_f = edges[('user', 'runs', 'process')]
                e_src.append(src_i); e_dst.append(dst_i); e_f.append(edge_f)

            if src_ip and user:
                src_i = registry.get_or_add('ip', src_ip)
                dst_i = registry.get_or_add('user', user)
                e_src, e_dst, e_f = edges[('ip', 'authenticates', 'user')]
                e_src.append(src_i); e_dst.append(dst_i); e_f.append(edge_f)

            if proc and fpath:
                src_i = registry.get_or_add('process', proc)
                dst_i = registry.get_or_add('file', fpath)
                e_src, e_dst, e_f = edges[('process', 'opens', 'file')]
                e_src.append(src_i); e_dst.append(dst_i); e_f.append(edge_f)

        # ── Assign node feature matrices ──
        n_ip   = registry.size('ip')
        n_user = registry.size('user')
        n_file = registry.size('file')
        n_proc = registry.size('process')

        if n_ip == 0 and n_user == 0:
            return None   # empty snapshot

        def _stack(feat_dict: Dict, n: int, dim: int) -> torch.Tensor:
            rows = []
            for i in range(n):
                rows.append(feat_dict.get(i, torch.zeros(dim)))
            return torch.stack(rows) if rows else torch.zeros(0, dim)

        data['ip'].x      = _stack(ip_feats,      n_ip,   IP_FEAT_DIM)
        data['user'].x    = _stack(user_feats,    n_user, USER_FEAT_DIM)
        data['file'].x    = _stack(file_feats,    n_file, FILE_FEAT_DIM)
        data['process'].x = _stack(process_feats, n_proc, PROCESS_FEAT_DIM)

        # Count of events per entity (degree proxy)
        data['ip'].num_events      = torch.tensor([n_ip])
        data['user'].num_events    = torch.tensor([n_user])

        # Timestamp metadata
        data.t_start = t_start
        data.t_end   = t_end

        # ── Assign edge indices & features ──
        for (src_type, rel, dst_type), (e_src, e_dst, e_f) in edges.items():
            if e_src:
                data[src_type, rel, dst_type].edge_index = torch.tensor(
                    [e_src, e_dst], dtype=torch.long
                )
                data[src_type, rel, dst_type].edge_attr = torch.stack(e_f)
            else:
                data[src_type, rel, dst_type].edge_index = torch.zeros(2, 0, dtype=torch.long)
                data[src_type, rel, dst_type].edge_attr  = torch.zeros(0, EDGE_FEAT_DIM)

        return data


# ──────────────────────────────────────────────
# Synthetic data generator (for testing / demo)
# ──────────────────────────────────────────────

class SyntheticAttackGenerator:
    """
    Generates realistic synthetic attack sequences for training & evaluation.

    Attack stages modelled:
      1. Reconnaissance  — port scans, ping sweeps
      2. Initial access  — brute force / phishing login
      3. Lateral movement
      4. Honeytoken access
      5. Privilege escalation
      6. Exfiltration
    """

    ATTACK_IPS   = [f'192.168.{i}.{j}' for i in range(1, 5) for j in range(1, 10)]
    BENIGN_IPS   = [f'10.0.{i}.{j}'    for i in range(0, 3) for j in range(1, 20)]
    USERS        = ['admin', 'serviceacct', 'jdoe', 'svc_backup', 'webserver', 'root']
    FILES        = [
        '/etc/passwd', '/etc/shadow', '/root/.ssh/id_rsa',
        '/honeytoken/credentials.txt', '/var/log/auth.log',
        '/home/jdoe/documents/budget.xlsx', '/opt/app/config.json',
    ]
    PROCESSES    = ['nmap', 'powershell', 'bash', 'python3', 'wget', 'curl', 'scp']
    ACTIONS      = ['connect', 'login', 'read', 'write', 'execute', 'scan',
                    'download', 'access_honeytoken', 'lateral_move', 'exfiltrate']

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

    def generate_attack_campaign(
        self,
        n_events: int = 500,
        attack_ratio: float = 0.3,
        t_start: float = 1_700_000_000.0,
        campaign_duration_sec: int = 3600,
    ) -> Tuple[List[Dict], List[int]]:
        """
        Returns (events, labels) where label 1 = attack event, 0 = benign.
        """
        events, labels = [], []
        n_attack = int(n_events * attack_ratio)
        n_benign = n_events - n_attack

        # Benign events — random background traffic
        for _ in range(n_benign):
            t = t_start + self.rng.uniform(0, campaign_duration_sec)
            events.append({
                'timestamp':   t,
                'src_ip':      self.rng.choice(self.BENIGN_IPS),
                'dst_ip':      self.rng.choice(self.BENIGN_IPS),
                'user':        self.rng.choice(self.USERS[2:]),
                'file_path':   self.rng.choice(self.FILES[4:]),
                'process':     self.rng.choice(['bash', 'python3']),
                'action':      self.rng.choice(['connect', 'read', 'login']),
                'port':        int(self.rng.choice([22, 80, 443, 3306])),
                'bytes':       int(self.rng.integers(100, 50_000)),
                'duration_ms': float(self.rng.uniform(10, 2000)),
                'success':     True,
            })
            labels.append(0)

        # Attack events — staged campaign
        attacker_ip  = self.rng.choice(self.ATTACK_IPS)
        pivot_ip     = self.rng.choice(self.BENIGN_IPS)
        attack_start = t_start + campaign_duration_sec * 0.2

        stage_events = self._generate_attack_stages(
            attacker_ip, pivot_ip, attack_start, n_attack
        )
        events.extend(stage_events)
        labels.extend([1] * len(stage_events))

        # Shuffle
        combined = list(zip(events, labels))
        self.rng.shuffle(combined)
        events, labels = zip(*combined)
        return list(events), list(labels)

    def _generate_attack_stages(
        self, attacker: str, pivot: str, t0: float, n: int
    ) -> List[Dict]:
        stages, t = [], t0
        per_stage = max(1, n // 6)

        # Stage 1: Recon
        targets = self.rng.choice(self.BENIGN_IPS, size=per_stage, replace=True)
        for tgt in targets:
            stages.append({
                'timestamp': t, 'src_ip': attacker, 'dst_ip': tgt,
                'user': '', 'file_path': '', 'process': 'nmap',
                'action': 'scan', 'port': int(self.rng.choice([22, 80, 443, 3389, 8080])),
                'bytes': 64, 'duration_ms': 50.0, 'success': bool(self.rng.random() > 0.5),
            })
            t += self.rng.uniform(1, 10)

        # Stage 2: Brute force
        for _ in range(per_stage):
            stages.append({
                'timestamp': t, 'src_ip': attacker, 'dst_ip': pivot,
                'user': 'admin', 'file_path': '', 'process': 'hydra',
                'action': 'login', 'port': 22, 'bytes': 200,
                'duration_ms': 500.0, 'success': bool(self.rng.random() > 0.8),
            })
            t += self.rng.uniform(0.5, 3)

        # Stage 3: Lateral movement
        lateral_targets = self.rng.choice(self.BENIGN_IPS, size=per_stage, replace=True)
        for tgt in lateral_targets:
            stages.append({
                'timestamp': t, 'src_ip': pivot, 'dst_ip': tgt,
                'user': 'admin', 'file_path': '', 'process': 'ssh',
                'action': 'lateral_move', 'port': 22, 'bytes': 1024,
                'duration_ms': 200.0, 'success': True,
            })
            t += self.rng.uniform(5, 30)

        # Stage 4: Honeytoken access — HIGH SIGNAL
        for _ in range(max(1, per_stage // 2)):
            stages.append({
                'timestamp': t, 'src_ip': pivot,
                'dst_ip': self.rng.choice(self.BENIGN_IPS),
                'user': 'admin', 'file_path': '/honeytoken/credentials.txt',
                'process': 'cat', 'action': 'access_honeytoken',
                'port': 0, 'bytes': 512, 'duration_ms': 10.0, 'success': True,
            })
            t += self.rng.uniform(1, 5)

        # Stage 5: Privilege escalation
        for _ in range(per_stage):
            stages.append({
                'timestamp': t, 'src_ip': pivot,
                'dst_ip': self.rng.choice(self.BENIGN_IPS),
                'user': 'serviceacct', 'file_path': '/etc/shadow',
                'process': 'sudo', 'action': 'escalate',
                'port': 0, 'bytes': 1024, 'duration_ms': 50.0, 'success': True,
            })
            t += self.rng.uniform(2, 10)

        # Stage 6: Exfiltration
        for _ in range(per_stage):
            stages.append({
                'timestamp': t, 'src_ip': pivot, 'dst_ip': attacker,
                'user': 'root', 'file_path': self.rng.choice(self.FILES[:3]),
                'process': 'scp', 'action': 'exfiltrate',
                'port': 443, 'bytes': int(self.rng.integers(500_000, 5_000_000)),
                'duration_ms': float(self.rng.uniform(5000, 60000)), 'success': True,
            })
            t += self.rng.uniform(10, 60)

        return stages