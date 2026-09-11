"""graph_builder.py - part of the GNN-BERT music-context project.
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



def cosine_matrix(x: np.ndarray) -> np.ndarray:
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return np.clip(xn @ xn.T, -1.0, 1.0)


def build_segment_graph(node_feats: np.ndarray, chroma_seg: np.ndarray, mfcc_seg: np.ndarray,
                        gcfg: GraphConfig) -> Dict[str, np.ndarray]:
    """Nodes = time segments. Edges = temporal adjacency + (MFCC/chroma cosine > tau) similarity edges."""
    n = node_feats.shape[0]
    src, dst, w, etype = [], [], [], []          # etype: 0 temporal, 1 similarity, 2 self-loop

    for i in range(n - 1):                        # temporal adjacency (both directions)
        src += [i, i + 1]; dst += [i + 1, i]; w += [1.0, 1.0]; etype += [0, 0]

    if n > 2:
        sim = 0.5 * cosine_matrix(chroma_seg) + 0.5 * cosine_matrix(mfcc_seg)
        np.fill_diagonal(sim, -np.inf)
        for i in range(n):
            for j in range(i + 2, n):             # skip already-connected temporal neighbours
                if sim[i, j] > gcfg.similarity_threshold:
                    src += [i, j]; dst += [j, i]
                    w += [float(sim[i, j])] * 2; etype += [1, 1]
        # guarantee a minimum of structure even when tau is never exceeded
        if not any(t == 1 for t in etype):
            k = min(gcfg.similarity_topk, n - 1)
            for i in range(n):
                for j in np.argsort(-sim[i])[:k]:
                    if abs(int(j) - i) > 1:
                        src += [i, int(j)]; dst += [int(j), i]
                        w += [float(max(sim[i, int(j)], 0.0))] * 2; etype += [1, 1]

    if gcfg.add_self_loops:
        for i in range(n):
            src.append(i); dst.append(i); w.append(1.0); etype.append(2)

    return {"x": node_feats.astype(np.float32),
            "edge_index": np.asarray([src, dst], dtype=np.int64).reshape(2, -1),
            "edge_weight": np.asarray(w, dtype=np.float32),
            "edge_type": np.asarray(etype, dtype=np.int64)}


def chord_templates() -> Tuple[np.ndarray, List[str]]:
    T, names = [], []
    for root in range(12):
        for triad, suffix in ((MAJ_TRIAD, ""), (MIN_TRIAD, "m")):
            v = np.zeros(12, dtype=np.float32)
            for s in triad:
                v[(root + s) % 12] = 1.0
            T.append(v / np.linalg.norm(v))
            names.append(f"{PITCH_CLASSES[root]}{suffix}")
    return np.stack(T), names


def _median_filter_1d(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or x.size < k:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.array([np.bincount(xp[i:i + k]).argmax() for i in range(x.size)], dtype=x.dtype)


def estimate_chord_path(feats: Dict[str, np.ndarray], gcfg: GraphConfig) -> Tuple[np.ndarray, np.ndarray]:
    chroma = feats["chroma"]
    if gcfg.beat_synchronous_chords and feats["beats"].size >= 2:
        units = librosa.util.sync(chroma, feats["beats"], aggregate=np.median)
    else:
        step = max(1, chroma.shape[1] // 64)
        units = np.stack([chroma[:, i:i + step].mean(1)
                          for i in range(0, chroma.shape[1] - step + 1, step)], axis=1)
    units = np.nan_to_num(units)
    un = units / (np.linalg.norm(units, axis=0, keepdims=True) + 1e-8)
    path = np.argmax(CHORD_TEMPLATES @ un, axis=0).astype(np.int64)
    return _median_filter_1d(path, gcfg.chord_median_filter), un


def build_chord_graph(path: np.ndarray, chroma_units: np.ndarray,
                      gcfg: GraphConfig) -> Dict[str, np.ndarray]:
    uniq = sorted(set(path.tolist()))
    if len(uniq) < gcfg.min_chord_nodes:                       # degenerate track -> add its relative
        extra = (uniq[0] + 1) % 24 if uniq else 0
        uniq = sorted(set(uniq + [extra]))
    idx = {c: i for i, c in enumerate(uniq)}
    n = len(uniq)

    x = np.zeros((n, CHORD_FEATURE_DIM), dtype=np.float32)
    counts = Counter(path.tolist())
    total = max(1, len(path))
    for c, i in idx.items():
        mask = (path == c)
        mean_chroma = chroma_units[:, mask].mean(1) if mask.any() else np.zeros(12, dtype=np.float32)
        root_oh = np.zeros(12, dtype=np.float32); root_oh[c // 2] = 1.0
        x[i] = np.concatenate([mean_chroma, root_oh,
                               [1.0 if c % 2 else 0.0],                 # 1 = minor quality
                               [counts.get(c, 0) / total],              # duration share
                               [math.log1p(counts.get(c, 0))]]).astype(np.float32)

    trans = Counter((int(a), int(b)) for a, b in zip(path[:-1], path[1:]) if a != b)
    src, dst, w = [], [], []
    for (a, b), cnt in trans.items():
        if a in idx and b in idx:
            src.append(idx[a]); dst.append(idx[b]); w.append(float(cnt))
    if not src:                                                # no transition observed
        for i in range(n):
            j = (i + 1) % n
            src.append(i); dst.append(j); w.append(1.0)
    w = np.asarray(w, dtype=np.float32)
    w = w / (w.max() + 1e-8)
    if gcfg.add_self_loops:
        src += list(range(n)); dst += list(range(n))
        w = np.concatenate([w, np.ones(n, dtype=np.float32)])

    return {"x": x,
            "edge_index": np.asarray([src, dst], dtype=np.int64).reshape(2, -1),
            "edge_weight": w.astype(np.float32),
            "chord_ids": np.asarray(uniq, dtype=np.int64),
            "chord_path": path.astype(np.int64)}


def export_graph_samples(keys: List[Any], cache_dir: Path, table: pd.DataFrame,
                         out_dir: Path, n: int, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for k in keys[:n]:
        rec = safe_torch_load(cache_dir / f"{k}.pt")
        torch.save({kk: vv for kk, vv in rec.items() if kk != "mel_cnn"},
                   out_dir / f"{prefix}_{k}.pt")
        row = table.loc[k]
        chord_ids = rec["chord_ids"].tolist()
        payload = {
            "id": str(k), "source": prefix,
            "label": str(row.get("genre_top", "")) or None,
            "aspects": list(row["aspect_list"])[:12] if "aspect_list" in table.columns else None,
            "tempo_bpm": round(float(rec["tempo"]), 2),
            "segment_graph": {
                "num_nodes": int(rec["seg_x"].shape[0]),
                "node_feature_dim": int(rec["seg_x"].shape[1]),
                "edge_index": rec["seg_edge_index"].tolist(),
                "edge_weight": [round(v, 4) for v in rec["seg_edge_weight"].tolist()],
                "edge_type": rec["seg_edge_type"].tolist(),
                "edge_type_legend": {"0": "temporal", "1": "similarity", "2": "self-loop"},
            },
            "chord_graph": {
                "num_nodes": len(chord_ids),
                "chords": [CHORD_NAMES[c] for c in chord_ids],
                "edge_index": rec["chord_edge_index"].tolist(),
                "edge_weight": [round(v, 4) for v in rec["chord_edge_weight"].tolist()],
                "chord_path": [CHORD_NAMES[c] for c in rec["chord_path"].tolist()[:64]],
            },
        }
        save_json(payload, out_dir / f"{prefix}_{k}.json")
        written += 1
    return written


def graph_stats(keys: List[Any], cache_dir: Path, n: int = 300) -> Dict[str, float]:
    ns, es, cn, ce, dens = [], [], [], [], []
    for k in keys[:n]:
        r = safe_torch_load(cache_dir / f"{k}.pt")
        v = int(r["seg_x"].shape[0]); e = int(r["seg_edge_index"].shape[1])
        ns.append(v); es.append(e); dens.append(e / max(1, v * (v - 1)))
        cn.append(int(r["chord_x"].shape[0])); ce.append(int(r["chord_edge_index"].shape[1]))
    return {"seg_nodes_mean": float(np.mean(ns)), "seg_edges_mean": float(np.mean(es)),
            "seg_density_mean": float(np.mean(dens)),
            "chord_nodes_mean": float(np.mean(cn)), "chord_edges_mean": float(np.mean(ce))}


def chord_walk(rec: Dict[str, Any], max_len: int = 8) -> List[Tuple[str, float]]:
    """Greedy highest-weight walk over the chord-transition graph, starting from the busiest chord."""
    ei, ew = rec["chord_edge_index"].numpy(), rec["chord_edge_weight"].numpy()
    ids = rec["chord_ids"].numpy()
    out_w = defaultdict(list)
    for (s, d), w in zip(ei.T, ew):
        if s != d:
            out_w[int(s)].append((int(d), float(w)))
    if not out_w:
        return [(CHORD_NAMES[int(ids[0])], 1.0)]
    start = max(out_w, key=lambda k: sum(w for _, w in out_w[k]))
    walk, cur, seen = [(CHORD_NAMES[int(ids[start])], 1.0)], start, {start}
    for _ in range(max_len - 1):
        cands = [(d, w) for d, w in out_w.get(cur, []) if d not in seen]
        if not cands:
            cands = out_w.get(cur, [])
        if not cands:
            break
        d, w = max(cands, key=lambda t: t[1])
        walk.append((CHORD_NAMES[int(ids[d])], round(w, 3)))
        seen.add(d); cur = d
    return walk
