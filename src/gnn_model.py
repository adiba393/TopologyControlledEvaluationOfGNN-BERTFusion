"""gnn_model.py - part of the GNN-BERT music-context project.
Auto-exported from the master notebook so the package and the executed code are identical.
"""
from __future__ import annotations
import ast, json, math, os, random, re, time, warnings
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    shape = (dim_size,) + tuple(src.shape[1:])
    out = torch.zeros(shape, dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    s = scatter_sum(src, index, dim_size)
    cnt = scatter_sum(torch.ones(index.shape[0], 1, dtype=src.dtype, device=src.device),
                      index, dim_size).clamp(min=1.0)
    return s / cnt


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    shape = (dim_size,) + tuple(src.shape[1:])
    out = torch.full(shape, float("-inf"), dtype=src.dtype, device=src.device)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    return torch.nan_to_num(out, neginf=0.0)


def scatter_softmax(score: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    m = scatter_max(score, index, dim_size)[index]
    e = torch.exp(score - m)
    denom = scatter_sum(e, index, dim_size)[index] + 1e-16
    return e / denom


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, n: int) -> torch.Tensor:
    return scatter_mean(x, batch, n)


def global_max_pool(x: torch.Tensor, batch: torch.Tensor, n: int) -> torch.Tensor:
    return scatter_max(x, batch, n)


class BuiltinSAGEConv(nn.Module):
    """GraphSAGE with mean aggregation - drop-in equivalent of torch_geometric.nn.SAGEConv."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=True)    # neighbourhood branch
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)   # root branch

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        agg = scatter_mean(x.index_select(0, src), dst, x.shape[0])
        return self.lin_l(agg) + self.lin_r(x)


class BuiltinGATConv(nn.Module):
    """Multi-head graph attention - drop-in equivalent of torch_geometric.nn.GATConv (concat heads)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.0,
                 negative_slope: float = 0.2):
        super().__init__()
        assert out_dim % heads == 0, "out_dim must be divisible by heads"
        self.h, self.d = heads, out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, self.d))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.d))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.slope, self.dropout = negative_slope, dropout
        nn.init.xavier_uniform_(self.att_src); nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_attention: bool = False):
        n = x.shape[0]
        h = self.lin(x).view(n, self.h, self.d)
        src, dst = edge_index[0], edge_index[1]
        a = ((h * self.att_src).sum(-1)[src] + (h * self.att_dst).sum(-1)[dst])
        a = F.leaky_relu(a, self.slope)
        alpha = scatter_softmax(a, dst, n)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = h[src] * alpha.unsqueeze(-1)
        out = scatter_sum(msg, dst, n).reshape(n, self.h * self.d) + self.bias
        return (out, alpha) if return_attention else out


def make_conv(kind: str, in_dim: int, out_dim: int, heads: int, dropout: float) -> nn.Module:
    if kind == "gat":
        if HAS_PYG:
            return PyGGATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
        return BuiltinGATConv(in_dim, out_dim, heads=heads, dropout=dropout)
    if HAS_PYG:
        return PyGSAGEConv(in_dim, out_dim, aggr="mean")
    return BuiltinSAGEConv(in_dim, out_dim)


class GNNBranch(nn.Module):
    """L message-passing layers with BatchNorm, residual connections and dropout."""

    def __init__(self, in_dim: int, hidden: int, layers: int, kind: str,
                 heads: int, dropout: float, jk: bool = False,
                 drop_edge: float = 0.0):
        super().__init__()
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.convs.append(make_conv(kind, d, hidden, heads, dropout))
            self.norms.append(nn.BatchNorm1d(hidden))
            d = hidden
        self.dropout = dropout
        self.jk, self.drop_edge = jk, drop_edge
        # v8: the JK path carries h^(0) - the raw node statistics plus the patch
        # embedding - straight to the readout, so the audio detail survives even
        # when message passing smooths h^(L) flat.
        self.in_proj = nn.Linear(in_dim, hidden) if jk else None
        self.out_dim = hidden * (layers + 1) if jk else hidden

    def _sparsify(self, edge_index: torch.Tensor) -> torch.Tensor:
        """DropEdge: shorten the effective receptive field during training."""
        if not (self.training and self.drop_edge > 0):
            return edge_index
        keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= self.drop_edge
        return edge_index if int(keep.sum()) == 0 else edge_index[:, keep]

    def forward(self, g: GraphBatch) -> torch.Tensor:
        h, ei = g.x, self._sparsify(g.edge_index)
        outs = [self.in_proj(g.x)] if self.jk else []
        for i, (conv, bn) in enumerate(zip(self.convs, self.norms)):
            h_new = F.relu(bn(conv(h, ei)))
            h = h_new + h if (i > 0 and h_new.shape == h.shape) else h_new
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.jk:
                outs.append(h)
        return torch.cat(outs, dim=-1) if self.jk else h


class DualGNNEncoder(nn.Module):
    """Segment branch + chord branch -> concatenated mean/max readout -> graph embedding g."""

    def __init__(self, seg_dim: int, chord_dim: int, mcfg: ModelConfig, use_chord: bool = True):
        super().__init__()
        self.use_chord = use_chord
        self.patch_enc = None
        if getattr(mcfg, "use_segment_patch_encoder", False):
            self.patch_enc = SegmentPatchEncoder(mcfg.patch_embed_dim, mcfg.gnn_dropout,
                                                 getattr(mcfg, "patch_pool", (1, 1)))
            seg_dim = seg_dim + self.patch_enc.out_dim
        _jk = bool(getattr(mcfg, "gnn_jumping_knowledge", False))
        _de = float(getattr(mcfg, "drop_edge", 0.0))
        self.seg = GNNBranch(seg_dim, mcfg.gnn_hidden, mcfg.gnn_layers, mcfg.gnn_type,
                             mcfg.gnn_heads, mcfg.gnn_dropout, jk=_jk, drop_edge=_de)
        if use_chord:
            self.chord = GNNBranch(chord_dim, mcfg.gnn_hidden, max(2, mcfg.gnn_layers - 1),
                                   mcfg.gnn_type, mcfg.gnn_heads, mcfg.gnn_dropout,
                                   jk=_jk, drop_edge=_de)
        # v8: readout width now follows the branch (JK widens it) and the number
        # of pooling statistics (mean/max, plus std when enabled).
        self._nstat = 3 if getattr(mcfg, "readout_std", False) else 2
        readout = self.seg.out_dim * self._nstat
        if use_chord:
            readout += self.chord.out_dim * self._nstat
        self.proj = nn.Sequential(nn.Linear(readout, mcfg.graph_embed_dim), nn.ReLU(),
                                  nn.Dropout(mcfg.gnn_dropout),
                                  nn.Linear(mcfg.graph_embed_dim, mcfg.graph_embed_dim))
        self.out_dim = mcfg.graph_embed_dim
        self.last_node_embeddings: Optional[torch.Tensor] = None

    def _readout(self, h: torch.Tensor, g: GraphBatch) -> torch.Tensor:
        mean = global_mean_pool(h, g.batch, g.num_graphs)
        parts = [mean, global_max_pool(h, g.batch, g.num_graphs)]
        if self._nstat == 3:
            # v8: per-graph standard deviation. When message passing collapses
            # the nodes, mean and max become the same vector but std still
            # reports how much variation is left - and it is exactly the signal
            # the v7 coherence score showed was being destroyed.
            sq = global_mean_pool(h * h, g.batch, g.num_graphs)
            parts.append((sq - mean * mean).clamp_min(1e-6).sqrt())
        return torch.cat(parts, dim=-1)

    def forward(self, seg: GraphBatch, chord: Optional[GraphBatch] = None) -> torch.Tensor:
        if self.patch_enc is not None:
            if seg.mel is None:
                raise RuntimeError(
                    "use_segment_patch_encoder=True but the cache has no 'seg_mel'. "
                    "Delete data/processed/fma_cache* and musiccaps_cache, then re-run "
                    "the feature cell so the patches get written.")
            mel = spec_augment(seg.mel, CFG.train, self.training)
            seg = GraphBatch(torch.cat([seg.x, self.patch_enc(mel)], dim=-1),
                             seg.edge_index, seg.edge_weight, seg.batch,
                             seg.num_graphs, seg.mel)
        hs = self.seg(seg)
        self.last_node_embeddings = hs
        parts = [self._readout(hs, seg)]
        if self.use_chord and chord is not None:
            parts.append(self._readout(self.chord(chord), chord))
        return self.proj(torch.cat(parts, dim=-1))


class GNNClassifier(nn.Module):
    """Task 2:  g = MEANPOOL(h^{(L)});  y_hat = head(g)."""

    def __init__(self, seg_dim: int, chord_dim: int, n_out: int, mcfg: ModelConfig,
                 use_chord: bool = True):
        super().__init__()
        self.encoder = DualGNNEncoder(seg_dim, chord_dim, mcfg, use_chord)
        self.head = nn.Sequential(nn.Dropout(mcfg.gnn_dropout),
                                  nn.Linear(self.encoder.out_dim, n_out))

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        g = self.encoder(batch["seg"], batch.get("chord"))
        return self.head(g)


class MelCNN(nn.Module):
    """Baseline B2: 4-block 2-D CNN on the 128 x 512 log-mel patch (no graph, no text)."""

    def __init__(self, n_out: int, channels: Tuple[int, ...] = (32, 64, 128, 256),
                 dropout: float = 0.3, multilabel: bool = False):
        super().__init__()
        blocks, c_in = [], 1
        for c in channels:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
                       nn.MaxPool2d(2), nn.Dropout2d(dropout * 0.5)]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(c_in, n_out))
        self.multilabel = multilabel

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.head(self.features(
            spec_augment(batch["mel"], CFG.train, self.training)))
