"""
Attack Behaviour Graph Visualizer.

Generates interactive-style graphs showing:
  - Nodes coloured by anomaly score (green → red heat map)
  - Edges coloured by relation type
  - Attacker paths highlighted
  - Temporal evolution across snapshots
"""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for server/script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import networkx as nx
from typing import Dict, List, Optional, Tuple
from torch_geometric.data import HeteroData


# Custom colour map: low anomaly = steel blue → high = crimson
THREAT_CMAP = LinearSegmentedColormap.from_list(
    'threat', ['#2196F3', '#FFC107', '#F44336'], N=256
)

NODE_SHAPES  = {'ip': 'o', 'user': 's', 'file': 'D', 'process': '^'}
NODE_COLORS_BASE = {
    'ip':      '#90CAF9',
    'user':    '#A5D6A7',
    'file':    '#FFCC80',
    'process': '#CE93D8',
}
EDGE_COLORS = {
    'connects_to':   '#78909C',
    'scans':         '#EF5350',
    'accesses':      '#FFA726',
    'runs':          '#AB47BC',
    'authenticates': '#26A69A',
    'opens':         '#8D6E63',
    'lateral_moves': '#E53935',
}


def _build_nx_graph(
    snap:          HeteroData,
    anomaly_scores: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[nx.DiGraph, Dict]:
    """
    Convert a PyG HeteroData snapshot into a NetworkX DiGraph.
    Node IDs are strings like 'ip_0', 'user_3', etc.

    Returns (graph, node_meta) where node_meta has display attributes.
    """
    G        = nx.DiGraph()
    node_meta = {}

    for ntype in ['ip', 'user', 'file', 'process']:
        if not hasattr(snap[ntype], 'x'):
            continue
        n = snap[ntype].x.size(0)
        scores = anomaly_scores.get(ntype, np.zeros(n)) if anomaly_scores else np.zeros(n)

        for i in range(n):
            node_id = f'{ntype}_{i}'
            score   = float(scores[i]) if i < len(scores) else 0.0
            G.add_node(node_id)
            node_meta[node_id] = {
                'type':   ntype,
                'index':  i,
                'score':  score,
                'color':  THREAT_CMAP(score),
                'shape':  NODE_SHAPES[ntype],
                'size':   300 + score * 800,   # bigger = more suspicious
                'label':  f'{ntype[0].upper()}{i}',
            }

    for (src_type, rel, dst_type) in snap.edge_types:
        ei = snap[src_type, rel, dst_type].edge_index
        if ei.size(1) == 0:
            continue
        for e in range(ei.size(1)):
            src_id = f'{src_type}_{ei[0, e].item()}'
            dst_id = f'{dst_type}_{ei[1, e].item()}'
            if src_id in G and dst_id in G:
                G.add_edge(src_id, dst_id, relation=rel, color=EDGE_COLORS.get(rel, '#90A4AE'))

    return G, node_meta


def visualize_snapshot(
    snap:            HeteroData,
    anomaly_scores:  Optional[Dict[str, np.ndarray]] = None,
    title:           str = 'Attack Behaviour Graph',
    save_path:       Optional[str] = None,
    figsize:         Tuple[int, int] = (14, 10),
    highlight_edges: Optional[List[Tuple[str, str]]] = None,
) -> plt.Figure:
    """
    Render a single graph snapshot as a matplotlib figure.

    Args:
        snap:            HeteroData object
        anomaly_scores:  {ntype: np.ndarray} of per-node scores in [0,1]
        title:           figure title
        save_path:       if set, saves PNG to this path
        highlight_edges: list of (src_id, dst_id) to draw in bright red
    """
    G, meta = _build_nx_graph(snap, anomaly_scores)

    if len(G.nodes) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Empty graph', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=figsize, facecolor='#0F1117')
    ax.set_facecolor('#0F1117')

    # Layout: spring with k tuning
    try:
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
    except Exception:
        pos = nx.random_layout(G, seed=42)

    # Draw each node type separately (different shapes)
    for ntype in ['ip', 'user', 'file', 'process']:
        nodes_of_type = [n for n, d in meta.items() if d['type'] == ntype]
        if not nodes_of_type:
            continue
        node_colors = [meta[n]['color'] for n in nodes_of_type]
        node_sizes  = [meta[n]['size']  for n in nodes_of_type]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes_of_type,
            node_color=node_colors,
            node_size=node_sizes,
            node_shape=NODE_SHAPES[ntype],
            ax=ax, alpha=0.95,
        )

    # Draw edges
    normal_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if not (highlight_edges and (u, v) in highlight_edges)]
    highlight_set = set(highlight_edges) if highlight_edges else set()
    attack_edges  = [(u, v) for u, v in G.edges() if (u, v) in highlight_set]

    # Group normal edges by relation colour
    relation_groups: Dict[str, List] = {}
    for u, v, d in G.edges(data=True):
        rel = d.get('relation', 'connects_to')
        if (u, v) not in highlight_set:
            relation_groups.setdefault(rel, []).append((u, v))

    for rel, elist in relation_groups.items():
        nx.draw_networkx_edges(
            G, pos, edgelist=elist,
            edge_color=EDGE_COLORS.get(rel, '#90A4AE'),
            ax=ax, alpha=0.5,
            arrows=True, arrowsize=15, arrowstyle='-|>',
            width=1.2,
            connectionstyle='arc3,rad=0.08',
        )

    if attack_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=attack_edges,
            edge_color='#FF1744', ax=ax, alpha=0.9,
            arrows=True, arrowsize=20, arrowstyle='-|>',
            width=2.5,
            connectionstyle='arc3,rad=0.08',
        )

    # Labels
    labels = {n: meta[n]['label'] for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax,
        font_size=7, font_color='white', font_weight='bold',
    )

    # ── Legend ──
    legend_elements = [
        mpatches.Patch(color=NODE_COLORS_BASE['ip'],      label='IP node'),
        mpatches.Patch(color=NODE_COLORS_BASE['user'],    label='User node'),
        mpatches.Patch(color=NODE_COLORS_BASE['file'],    label='File node'),
        mpatches.Patch(color=NODE_COLORS_BASE['process'], label='Process node'),
    ]
    for rel, col in EDGE_COLORS.items():
        legend_elements.append(Line2D([0], [0], color=col, linewidth=2, label=rel.replace('_', ' ')))
    if attack_edges:
        legend_elements.append(Line2D([0], [0], color='#FF1744', linewidth=2.5, label='Attack path'))

    ax.legend(
        handles=legend_elements,
        loc='upper left', fontsize=8,
        facecolor='#1E222D', edgecolor='#444',
        labelcolor='white', framealpha=0.85,
    )

    # Colorbar for anomaly score
    sm = plt.cm.ScalarMappable(cmap=THREAT_CMAP, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Anomaly score', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    ax.set_title(title, color='white', fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F1117')

    return fig


def visualize_attack_campaign(
    snapshots:       List[HeteroData],
    results:         List[Dict],
    save_dir:        str = 'output/graphs',
    max_snapshots:   int = 6,
):
    """
    Render multiple snapshots from an attack campaign in a grid layout.
    Shows temporal evolution of the attack graph.

    Args:
        snapshots: list of HeteroData objects
        results:   inference output from AttackBehaviourTGN.infer_sequence()
        save_dir:  directory to save individual and combined figures
    """
    os.makedirs(save_dir, exist_ok=True)
    n = min(len(snapshots), max_snapshots)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 5.5), facecolor='#0F1117')
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i in range(n):
        snap   = snapshots[i]
        result = results[i] if i < len(results) else {}
        ax     = axes[i]

        anomaly_scores = {
            k: v.flatten()
            for k, v in result.get('node_scores', {}).items()
        }

        G, meta = _build_nx_graph(snap, anomaly_scores)
        ax.set_facecolor('#0F1117')

        if len(G.nodes) == 0:
            ax.text(0.5, 0.5, 'Empty', ha='center', va='center', color='white')
            ax.axis('off')
            continue

        try:
            pos = nx.spring_layout(G, k=2.0, seed=42)
        except Exception:
            pos = nx.random_layout(G, seed=42)

        for ntype in ['ip', 'user', 'file', 'process']:
            nodes_t = [nd for nd, d in meta.items() if d['type'] == ntype]
            if not nodes_t:
                continue
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes_t, ax=ax,
                node_color=[meta[nd]['color'] for nd in nodes_t],
                node_size=[meta[nd]['size']  for nd in nodes_t],
                node_shape=NODE_SHAPES[ntype], alpha=0.9,
            )

        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=0.4,
            edge_color='#78909C', arrows=True, arrowsize=12,
            width=1.0, connectionstyle='arc3,rad=0.1',
        )
        nx.draw_networkx_labels(
            G, pos, {nd: meta[nd]['label'] for nd in G.nodes()},
            ax=ax, font_size=6, font_color='white',
        )

        behaviour  = result.get('behaviour', 'unknown')
        confidence = result.get('behaviour_conf', 0.0)
        anomaly    = result.get('anomaly_mean', 0.0)
        ax.set_title(
            f'T{i+1}  |  {behaviour}  ({confidence:.0%})\n'
            f'Anomaly: {anomaly:.3f}  |  Nodes: {len(G.nodes)}  Edges: {len(G.edges)}',
            color='white', fontsize=9, pad=6,
        )
        ax.axis('off')

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Attack Behaviour Graph — Temporal Evolution', color='white', fontsize=14, y=1.01)
    plt.tight_layout()
    campaign_path = os.path.join(save_dir, 'campaign_evolution.png')
    fig.savefig(campaign_path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
    print(f'[Visualizer] Saved campaign grid → {campaign_path}')

    plt.close('all')
    return campaign_path


def visualize_anomaly_timeline(
    results:   List[Dict],
    save_path: str = 'output/graphs/anomaly_timeline.png',
) -> str:
    """
    Plot anomaly score over time + behaviour predictions as colour-coded bands.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    times      = [r.get('t_start', i * 60) for i, r in enumerate(results)]
    anomalies  = [r.get('anomaly_mean', 0.0) for r in results]
    maxes      = [r.get('anomaly_max',  0.0) for r in results]
    behaviours = [r.get('behaviour', 'benign') for r in results]

    BHVR_COLORS = {
        'benign':               '#4CAF50',
        'reconnaissance':       '#2196F3',
        'lateral_movement':     '#FF9800',
        'privilege_escalation': '#9C27B0',
        'exfiltration':         '#F44336',
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), facecolor='#0F1117',
                                    gridspec_kw={'height_ratios': [3, 1]})
    ax1.set_facecolor('#0F1117')
    ax2.set_facecolor('#0F1117')

    ax1.fill_between(times, anomalies, alpha=0.3, color='#EF5350')
    ax1.plot(times, anomalies, color='#EF5350', linewidth=2, label='Mean anomaly')
    ax1.plot(times, maxes,     color='#FF7043', linewidth=1, linestyle='--', alpha=0.7, label='Max anomaly')
    ax1.axhline(0.5, color='#FFC107', linewidth=1.2, linestyle=':', label='Alert threshold')
    ax1.axhline(0.8, color='#FF1744', linewidth=1.2, linestyle=':', label='Critical threshold')

    ax1.set_ylabel('Anomaly score', color='white')
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(colors='white')
    ax1.spines['bottom'].set_color('#444')
    ax1.spines['left'].set_color('#444')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(fontsize=9, facecolor='#1E222D', labelcolor='white', edgecolor='#444')
    ax1.set_title('Anomaly Score Timeline', color='white', fontsize=12)

    # Behaviour band
    for i, (t, b) in enumerate(zip(times, behaviours)):
        color = BHVR_COLORS.get(b, '#78909C')
        t_next = times[i + 1] if i + 1 < len(times) else t + 60
        ax2.barh(0, t_next - t, left=t, color=color, height=0.8, alpha=0.85)

    # Behaviour legend
    bhvr_patches = [mpatches.Patch(color=c, label=b.replace('_', ' '))
                    for b, c in BHVR_COLORS.items()]
    ax2.legend(handles=bhvr_patches, loc='center left', fontsize=8,
               facecolor='#1E222D', labelcolor='white', edgecolor='#444',
               bbox_to_anchor=(1.01, 0.5))
    ax2.set_yticks([])
    ax2.set_xlabel('Time (s)', color='white')
    ax2.tick_params(colors='white')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.set_facecolor('#0F1117')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0F1117')
    plt.close('all')
    print(f'[Visualizer] Saved anomaly timeline → {save_path}')
    return save_path