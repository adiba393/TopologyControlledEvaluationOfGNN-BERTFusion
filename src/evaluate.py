"""evaluate.py - part of the GNN-BERT music-context project.
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



def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                    grid: np.ndarray = np.arange(0.05, 0.96, 0.05)) -> np.ndarray:
    K = y_true.shape[1]
    th = np.full(K, 0.5, dtype=np.float32)
    for k in range(K):
        if y_true[:, k].sum() == 0:
            continue
        best, best_f1 = 0.5, -1.0
        for t in grid:
            f1 = f1_score(y_true[:, k], (y_prob[:, k] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best, best_f1 = float(t), f1
        th[k] = best
    return th


def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                       thresholds: Optional[np.ndarray] = None) -> Dict[str, float]:
    th = np.full(y_true.shape[1], 0.5, dtype=np.float32) if thresholds is None else thresholds
    y_pred = (y_prob >= th[None, :]).astype(int)
    valid = y_true.sum(0) > 0
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
    }
    if valid.any():
        aps = [average_precision_score(y_true[:, k], y_prob[:, k]) for k in np.where(valid)[0]]
        out["auc_pr_macro"] = float(np.mean(aps))
        out["auc_pr_micro"] = float(average_precision_score(y_true[:, valid].ravel(),
                                                            y_prob[:, valid].ravel()))
        try:
            out["roc_auc_macro"] = float(roc_auc_score(y_true[:, valid], y_prob[:, valid], average="macro"))
        except Exception:
            out["roc_auc_macro"] = float("nan")
    out["n_labels_scored"] = int(valid.sum())
    return out


def per_tag_report(y_true: np.ndarray, y_prob: np.ndarray, names: Sequence[str],
                   thresholds: np.ndarray) -> pd.DataFrame:
    y_pred = (y_prob >= thresholds[None, :]).astype(int)
    rows = []
    for k, name in enumerate(names):
        tp = int(((y_pred[:, k] == 1) & (y_true[:, k] == 1)).sum())
        fp = int(((y_pred[:, k] == 1) & (y_true[:, k] == 0)).sum())
        fn = int(((y_pred[:, k] == 0) & (y_true[:, k] == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        ap = float(average_precision_score(y_true[:, k], y_prob[:, k])) if y_true[:, k].sum() else float("nan")
        rows.append({"tag": name, "support": int(y_true[:, k].sum()), "threshold": float(thresholds[k]),
                     "precision": prec, "recall": rec, "f1": f1, "auc_pr": ap})
    return pd.DataFrame(rows).sort_values("support", ascending=False).reset_index(drop=True)


def single_label_metrics(y_true: np.ndarray, logits: np.ndarray, n_classes: int) -> Dict[str, float]:
    prob = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    pred = prob.argmax(1)
    oh = np.eye(n_classes)[y_true]
    out = {"accuracy": float((pred == y_true).mean()),
           "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
           "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0))}
    try:
        out["auc_pr_macro"] = float(np.mean([average_precision_score(oh[:, k], prob[:, k])
                                             for k in range(n_classes) if oh[:, k].sum() > 0]))
        out["roc_auc_macro"] = float(roc_auc_score(oh, prob, average="macro", multi_class="ovr"))
    except Exception:
        out["auc_pr_macro"] = float("nan"); out["roc_auc_macro"] = float("nan")
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(((y_true - y_pred) ** 2).mean()))
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) + 1e-12
    return {f"{prefix}mae": mae, f"{prefix}rmse": rmse, f"{prefix}r2": 1.0 - ss_res / ss_tot}


def retrieval_metrics(sim: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> Dict[str, float]:
    """`sim[i, j]` = score of query i against gallery j; the ground-truth match is the diagonal."""
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    ranks = np.array([int(np.where(order[i] == i)[0][0]) for i in range(n)]) + 1
    out = {f"R@{k}": float((ranks <= k).mean()) for k in ks}
    out["median_rank"] = float(np.median(ranks))
    out["mean_rank"] = float(ranks.mean())
    out["mrr"] = float((1.0 / ranks).mean())
    return out


def graph_coherence(h: torch.Tensor, edge_index: torch.Tensor, tau: float = 0.5) -> float:
    """S_graph = (1/|E|) * sum_{(i,j) in E} 1[cos(h_i, h_j) > tau]."""
    if edge_index.numel() == 0:
        return float("nan")
    hn = F.normalize(h, dim=-1)
    cos = (hn[edge_index[0]] * hn[edge_index[1]]).sum(-1)
    return float((cos > tau).float().mean().item())


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, kind: str = "multilabel") -> Dict[str, np.ndarray]:
    """Collect logits/targets over a loader. `kind` in {'multilabel','single','embed'}."""
    model.eval()
    L, Y, YS, Z = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
            out = model(batch)
        logits = out[0] if isinstance(out, tuple) else out
        L.append(logits.float().cpu())
        if "y" in batch:
            Y.append(batch["y"].float().cpu())
        if "y_single" in batch:
            YS.append(batch["y_single"].cpu())
        if getattr(model, "last_z", None) is not None:
            Z.append(model.last_z.float().cpu())
    out = {"logits": torch.cat(L).numpy()}
    if Y:
        out["y"] = torch.cat(Y).numpy()
    if YS:
        out["y_single"] = torch.cat(YS).numpy()
    if Z:
        out["z"] = torch.cat(Z).numpy()
    if kind == "multilabel":
        out["prob"] = 1.0 / (1.0 + np.exp(-out["logits"]))
    return out


def eval_multilabel(model: nn.Module, loader: DataLoader,
                    thresholds: Optional[np.ndarray] = None) -> Dict[str, float]:
    out = predict(model, loader, "multilabel")
    return multilabel_metrics(out["y"], out["prob"], thresholds)


def eval_genre(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    out = predict(model, loader, "single")
    return single_label_metrics(out["y_single"], out["logits"], len(FMA_GENRES_SMALL))
