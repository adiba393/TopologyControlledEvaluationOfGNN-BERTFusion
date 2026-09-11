"""fusion_model.py - part of the GNN-BERT music-context project.
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



class CrossAttentionFusion(nn.Module):
    """z = CONCAT(g, A H_text) with A = softmax(QK^T / sqrt(d)), Q = g W_Q, K = H_text W_K."""

    def __init__(self, g_dim: int, t_dim: int, d: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d % heads == 0
        self.h, self.dk, self.d = heads, d // heads, d
        self.q = nn.Linear(g_dim, d)
        self.k = nn.Linear(t_dim, d)
        self.v = nn.Linear(t_dim, d)
        self.o = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, g: torch.Tensor, H: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = H.shape
        q = self.q(g).view(B, 1, self.h, self.dk).transpose(1, 2)        # [B,h,1,dk]
        k = self.k(H).view(B, L, self.h, self.dk).transpose(1, 2)        # [B,h,L,dk]
        v = self.v(H).view(B, L, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)          # [B,h,1,L]
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min)
        A = torch.softmax(scores, dim=-1)
        self.last_attention = A.mean(1).squeeze(1).detach()              # [B,L] head-averaged
        ctx = (self.drop(A) @ v).transpose(1, 2).reshape(B, self.d)
        return self.norm(self.o(ctx))


class FusionModel(nn.Module):
    """Task 3. mode in {'xattn','concat','bert_only','gnn_only'} drives the required 4-way ablation."""

    def __init__(self, seg_dim: int, chord_dim: int, n_labels: int, cfg: Config,
                 mode: str = "xattn", use_emotion: bool = False, use_chord: bool = True):
        super().__init__()
        self.mode, self.use_emotion = mode, use_emotion
        m = cfg.model
        self.gnn = None if mode == "bert_only" else DualGNNEncoder(seg_dim, chord_dim, m, use_chord)
        self.text = None if mode == "gnn_only" else BertTextEncoder(cfg.text.bert_name,
                                                                    cfg.text.freeze_bert_layers)
        g_dim = self.gnn.out_dim if self.gnn is not None else 0
        t_dim = self.text.out_dim if self.text is not None else 0

        # v8: a learned scalar per modality, initialised neutral (softmax*2 =
        # 1,1). v7 cross-attention scored BELOW BERT-only because z always
        # carried g at full scale; the gate lets the trunk shrink a branch it
        # cannot use instead of being dragged down by it.
        self.mod_gate = (nn.Parameter(torch.zeros(2))
                         if getattr(m, "fusion_gate", False) and mode in ("xattn", "concat")
                         else None)
        self.mod_drop = float(getattr(cfg.train, "modality_dropout", 0.0)) \
            if mode in ("xattn", "concat") else 0.0

        if mode == "xattn":
            self.xattn = CrossAttentionFusion(g_dim, t_dim, m.fusion_dim, heads=4,
                                              dropout=m.fusion_dropout)
            z_dim = g_dim + m.fusion_dim                       # z = CONCAT(g, A H_text)
        elif mode == "concat":
            z_dim = g_dim + t_dim                              # early/late concat ablation
        else:
            z_dim = g_dim + t_dim                              # single-branch ablations

        self.trunk = nn.Sequential(nn.Linear(z_dim, m.fusion_dim), nn.ReLU(),
                                   nn.Dropout(m.fusion_dropout),
                                   nn.Linear(m.fusion_dim, m.fusion_dim), nn.ReLU())
        self.tag_head = nn.Linear(m.fusion_dim, n_labels)
        self.emotion_head = nn.Linear(m.fusion_dim, 2) if use_emotion else None
        self.last_z: Optional[torch.Tensor] = None

    def _combine(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Gate the two modality vectors, then optionally drop one of them.

        Modality dropout zeroes exactly one branch at a time (never both), which
        stops the trunk from learning to ignore the graph outright - the failure
        mode behind v7's fusion < BERT-only result.
        """
        if self.mod_gate is not None:
            w = torch.softmax(self.mod_gate, 0) * 2.0
            a, b = a * w[0], b * w[1]
        if self.training and self.mod_drop > 0:
            r = torch.rand(a.shape[0], 1, device=a.device)
            a = a * (r >= self.mod_drop).to(a.dtype)
            b = b * ((r < self.mod_drop) | (r >= 2 * self.mod_drop)).to(b.dtype)
        return torch.cat([a, b], dim=-1)

    def embed(self, batch: Dict[str, Any]) -> torch.Tensor:
        g = t = H = None
        if self.gnn is not None:
            g = self.gnn(batch["seg"], batch.get("chord"))
        if self.text is not None:
            H, t = self.text(batch["input_ids"], batch["attention_mask"])
        if self.mode == "xattn":
            z = self._combine(g, self.xattn(g, H, batch["attention_mask"]))
        elif self.mode == "concat":
            z = self._combine(g, t)
        elif self.mode == "gnn_only":
            z = g
        else:
            z = t
        self.last_z = z
        return self.trunk(z)

    def forward(self, batch: Dict[str, Any]):
        f = self.embed(batch)
        logits = self.tag_head(f)
        return (logits, self.emotion_head(f)) if self.emotion_head is not None else logits


def load_deam_targets(deam_dir: str, keys: Sequence[Any]) -> Optional[np.ndarray]:
    """Return an [N, 2] array of (valence, arousal) in [1, 9] when DEAM annotations are available.
    FMA and DEAM do not share track ids, so this is only populated when you supply your own mapping in
    `deam_dir/fma_to_deam.json` ({fma_track_id: deam_song_id})."""
    d = Path(deam_dir)
    ann = d / "annotations" / "annotations averaged per song" / "song_level" / \
          "static_annotations_averaged_songs_1_2000.csv"
    mapping = d / "fma_to_deam.json"
    if not (ann.exists() and mapping.exists()):
        return None
    df = pd.read_csv(ann).set_index("song_id")
    m = {int(k): int(v) for k, v in load_json(mapping).items()}
    out = np.full((len(keys), 2), np.nan, dtype=np.float32)
    for i, k in enumerate(keys):
        sid = m.get(int(k))
        if sid in df.index:
            out[i] = [float(df.loc[sid, " valence_mean"]), float(df.loc[sid, " arousal_mean"])]
    return out if np.isfinite(out).any() else None


def fusion_loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    out = model(batch)
    if isinstance(out, tuple):
        logits, emo = out
        loss = F.binary_cross_entropy_with_logits(logits, batch["y"])
        va = batch.get("va")
        if va is not None:
            m = torch.isfinite(va).all(dim=1)
            if m.any():
                loss = (loss
                        + CFG.train.alpha_valence * F.mse_loss(emo[m, 0], va[m, 0])
                        + CFG.train.beta_arousal * F.mse_loss(emo[m, 1], va[m, 1]))
        return loss
    return F.binary_cross_entropy_with_logits(out, batch["y"])


def run_fusion_variant(mode: str, epochs: int, loaders: Dict[str, DataLoader],
                       n_labels: int, tag: str, use_emotion: bool = False,
                       verbose: int = 1) -> Tuple[nn.Module, Dict[str, float], List[Dict[str, float]], np.ndarray]:
    set_seed(CFG.train.seed)
    model = FusionModel(SEG_DIM, CHORD_DIM, n_labels, CFG, mode=mode, use_emotion=use_emotion)
    tr = Trainer(model, fusion_loss_fn, CFG.train, monitor="macro_f1", mode="max", name=tag)
    hist = tr.fit(loaders["training"], loaders["validation"], epochs,
                  eval_fn=eval_multilabel, verbose=verbose)
    v = predict(model, loaders["validation"], "multilabel")
    th = tune_thresholds(v["y"], v["prob"])
    te = predict(model, loaders["test"], "multilabel")
    m = multilabel_metrics(te["y"], te["prob"], th)
    print(f"  {tag:<24} macro_f1={m['macro_f1']:.4f}  micro_f1={m['micro_f1']:.4f}  "
          f"auc_pr={m['auc_pr_macro']:.4f}")
    return model, m, hist, th


@torch.no_grad()
def case_studies(model: FusionModel, table: pd.DataFrame, idx: np.ndarray, texts: Sequence[str],
                 y: np.ndarray, space: LabelSpace, thresholds: np.ndarray,
                 cache_dir: Path, n: int = 3) -> pd.DataFrame:
    model.eval()
    pick = list(idx[:n])
    keys = [table.index[i] for i in pick]
    ds = MusicDataset(keys=keys, cache_dir=cache_dir, texts=[texts[i] for i in pick],
                      y_multi=y[pick], tokenizer=TOKENIZER,
                      max_length=CFG.text.max_length_metadata,
                      modalities=FUSION_MODALITIES, feature_norm=FMA_FN)
    batch = move_batch(make_collate(FUSION_MODALITIES)([ds[i] for i in range(len(ds))]), DEVICE)
    logits = model(batch)
    logits = logits[0] if isinstance(logits, tuple) else logits
    prob = torch.sigmoid(logits).float().cpu().numpy()
    attn = model.xattn.last_attention.float().cpu().numpy()          # [n, L]

    rows = []
    fig, axes = plt.subplots(n, 1, figsize=(13, 1.5 * n + 1.0))
    axes = np.atleast_1d(axes)
    for r, (i, k) in enumerate(zip(pick, keys)):
        rec = safe_torch_load(cache_dir / f"{k}.pt")
        walk = chord_walk(rec)
        ids = batch["input_ids"][r].cpu().numpy()
        mask = batch["attention_mask"][r].cpu().numpy().astype(bool)
        toks = TOKENIZER.convert_ids_to_tokens(ids[mask])[:40]
        w = attn[r][mask][:len(toks)]; w = w / (w.max() + 1e-9)
        axes[r].imshow(w[None, :], aspect="auto", cmap="Purples", vmin=0, vmax=1)
        axes[r].set_xticks(range(len(toks)))
        axes[r].set_xticklabels(toks, rotation=90, fontsize=6)
        axes[r].set_yticks([]); axes[r].grid(False)
        axes[r].set_title(f"case {r+1}: track {k} ({table.loc[k, 'genre_top']})  |  chord path: "
                          + " -> ".join(c for c, _ in walk), fontsize=7, loc="left")
        top_tok = [toks[j] for j in np.argsort(-w)[:6]]
        rows.append({
            "case": r + 1, "track_id": str(k), "genre": str(table.loc[k, "genre_top"]),
            "tempo_bpm": round(float(rec["tempo"]), 1),
            "chord_path": " -> ".join(f"{c}({v})" for c, v in walk),
            "n_chord_nodes": int(rec["chord_x"].shape[0]),
            "n_segment_nodes": int(rec["seg_x"].shape[0]),
            "text_view": texts[i][:120],
            "top_attended_tokens": ", ".join(top_tok),
            "true_tags": ", ".join(space.decode(y[i]))[:110],
            "pred_tags": ", ".join([space.names[c] for c in range(len(space))
                                    if prob[r, c] >= thresholds[c]])[:110],
            "pred_top5": ", ".join(f"{a}:{b:.2f}" for a, b in space.top_k(prob[r], 5)),
        })
    fig.suptitle("Task 3 - cross-attention from the graph query g onto the text tokens", y=1.02)
    savefig(fig, "08_task3_case_studies"); plt.show()
    return pd.DataFrame(rows)


@torch.no_grad()
def coherence_analysis(model: FusionModel, loader: DataLoader, taus=(0.3, 0.5, 0.7),
                       max_batches: int = 12) -> pd.DataFrame:
    model.eval()
    rows = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        batch = move_batch(batch, DEVICE)
        model.gnn(batch["seg"], batch.get("chord"))
        h = model.gnn.last_node_embeddings
        ei = batch["seg"].edge_index
        perm = torch.randperm(h.shape[0], device=h.device)          # random-rewiring null
        ei_rand = torch.stack([ei[0], perm[ei[1] % h.shape[0]]])
        for tau in taus:
            rows.append({"tau": tau, "observed": graph_coherence(h, ei, tau),
                         "random_null": graph_coherence(h, ei_rand, tau)})
    df = pd.DataFrame(rows).groupby("tau").mean().reset_index()
    df["lift"] = df["observed"] - df["random_null"]
    return df
