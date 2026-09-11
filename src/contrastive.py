"""contrastive.py - part of the GNN-BERT music-context project.
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



class DualEncoderContrastive(nn.Module):
    """Task 4: g_i = Norm(GNN(G_i)), t_i = Norm(BERT_CLS(caption_i)); InfoNCE over the batch."""

    def __init__(self, seg_dim: int, chord_dim: int, cfg: Config):
        super().__init__()
        m = cfg.model
        self.gnn = DualGNNEncoder(seg_dim, chord_dim, m, use_chord=True)
        self.text = BertTextEncoder(cfg.text.bert_name,
                                    getattr(cfg.text, "_freeze_override",
                                            cfg.text.freeze_bert_layers))
        self.gproj = nn.Sequential(nn.Linear(self.gnn.out_dim, m.projection_dim), nn.ReLU(),
                                   nn.Linear(m.projection_dim, m.projection_dim))
        self.tproj = nn.Sequential(nn.Linear(self.text.out_dim, m.projection_dim), nn.ReLU(),
                                   nn.Linear(m.projection_dim, m.projection_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / cfg.train.contrastive_temperature)))

    def encode_graph(self, batch: Dict[str, Any]) -> torch.Tensor:
        return F.normalize(self.gproj(self.gnn(batch["seg"], batch.get("chord"))), dim=-1)

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        _, t = self.text(input_ids, attention_mask)
        return F.normalize(self.tproj(t), dim=-1)

    def forward(self, batch: Dict[str, Any]):
        g = self.encode_graph(batch)
        t = self.encode_text(batch["input_ids"], batch["attention_mask"])
        return g, t, self.logit_scale.exp().clamp(max=100.0)


def info_nce(g: torch.Tensor, t: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Symmetric InfoNCE. `g2t` is the exact loss written in the brief; the reverse term is added
    because retrieval is evaluated in both directions."""
    logits = scale * g @ t.t()
    target = torch.arange(g.shape[0], device=g.device)
    l_g2t = F.cross_entropy(logits, target)
    l_t2g = F.cross_entropy(logits.t(), target)
    loss = 0.5 * (l_g2t + l_t2g)
    with torch.no_grad():
        acc = (logits.argmax(1) == target).float().mean().item()
    return loss, {"nce_g2t": l_g2t.item(), "nce_t2g": l_t2g.item(), "batch_acc": acc}


@torch.no_grad()
def embed_split(model: DualEncoderContrastive, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    G, T = [], []
    for batch in loader:
        batch = move_batch(batch, DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
            g, t, _ = model(batch)
        G.append(g.float().cpu()); T.append(t.float().cpu())
    return torch.cat(G).numpy(), torch.cat(T).numpy()


def eval_retrieval(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    G, T = embed_split(model, loader)
    sim = G @ T.T
    c2a = retrieval_metrics(sim.T)          # caption query -> audio gallery
    a2c = retrieval_metrics(sim)            # audio query   -> caption gallery
    out = {f"c2a_{k}": v for k, v in c2a.items()}
    out.update({f"a2c_{k}": v for k, v in a2c.items()})
    out["macro_f1"] = 0.5 * (c2a["R@5"] + a2c["R@5"])       # monitored quantity for early stopping
    return out


@torch.no_grad()
def zero_shot_tag_scores(model: DualEncoderContrastive, space: LabelSpace,
                         graph_emb: np.ndarray,
                         templates: Sequence[str] = ("a music clip that sounds {}",
                                                     "this song is {}",
                                                     "{} music")) -> np.ndarray:
    model.eval()
    protos = []
    for tag in space.names:
        enc = TOKENIZER([tpl.format(tag) for tpl in templates], padding=True, truncation=True,
                        max_length=32, return_tensors="pt").to(DEVICE)
        e = model.encode_text(enc["input_ids"], enc["attention_mask"]).float().mean(0)
        protos.append(F.normalize(e, dim=-1).cpu().numpy())
    P = np.stack(protos)                                    # [K, d]
    return graph_emb @ P.T                                  # cosine, both sides are L2-normalised


def aggregate_human_eval(path: Path, min_listeners: int = 5) -> Optional[Dict[str, Any]]:
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    df = df[pd.to_numeric(df["rating_1_to_5"], errors="coerce").notna()]
    if df.empty:
        return None
    df["rating_1_to_5"] = df["rating_1_to_5"].astype(float)
    per_listener = df.groupby("listener_id")["rating_1_to_5"].mean()
    return {"n_listeners": int(df["listener_id"].nunique()),
            "meets_minimum": bool(df["listener_id"].nunique() >= min_listeners),
            "mean_rating": float(df["rating_1_to_5"].mean()),
            "mean_rating_rank1": float(df.loc[df["retrieved_rank"] == 1, "rating_1_to_5"].mean()),
            "std_between_listeners": float(per_listener.std()),
            "per_listener_mean": per_listener.round(3).to_dict()}
