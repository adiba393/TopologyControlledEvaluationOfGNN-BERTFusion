"""train.py - part of the GNN-BERT music-context project.
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



def set_seed(seed: int) -> None:
    """Seed every random source used anywhere in the notebook."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = not FAST_MODE
    torch.backends.cudnn.benchmark = FAST_MODE


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    bert_p, head_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (bert_p if ("text.bert" in name or name.startswith("encoder.bert") or ".bert." in name)
         else head_p).append(p)
    groups = [g for g in ({"params": bert_p, "lr": cfg.lr_bert},
                          {"params": head_p, "lr": cfg.lr_head}) if g["params"]]
    return torch.optim.AdamW(groups, lr=cfg.lr_head, weight_decay=cfg.weight_decay)


def make_scheduler(opt: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.02, 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog))))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class Trainer:
    def __init__(self, model: nn.Module, loss_fn: Callable, cfg: TrainConfig,
                 monitor: str = "macro_f1", mode: str = "max", name: str = "model"):
        self.model = model.to(DEVICE)
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.monitor, self.mode, self.name = monitor, mode, name
        self.history: List[Dict[str, float]] = []
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_score = -math.inf if mode == "max" else math.inf

    def _better(self, s: float) -> bool:
        return s > self.best_score if self.mode == "max" else s < self.best_score

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader],
            epochs: int, eval_fn: Optional[Callable] = None, verbose: int = 1) -> List[Dict[str, float]]:
        opt = make_optimizer(self.model, self.cfg)
        sched = make_scheduler(opt, max(1, epochs * len(train_loader)), self.cfg.warmup_ratio)
        scaler = make_scaler(NEEDS_SCALER)   # bf16 needs no loss scaling
        bad = 0
        for ep in range(1, epochs + 1):
            self.model.train()
            tot, nb, t0 = 0.0, 0, time.time()
            for batch in train_loader:
                batch = move_batch(batch, DEVICE)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE, enabled=USE_AMP):
                    loss = self.loss_fn(self.model, batch)
                    if isinstance(loss, tuple):
                        loss = loss[0]
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                tot += float(loss.item()); nb += 1
            rec = {"epoch": ep, "train_loss": tot / max(1, nb), "seconds": time.time() - t0,
                   "lr": float(opt.param_groups[-1]["lr"])}
            if val_loader is not None and eval_fn is not None:
                rec.update({f"val_{k}": v for k, v in eval_fn(self.model, val_loader).items()
                            if isinstance(v, (int, float))})
                score = rec.get(f"val_{self.monitor}", -math.inf if self.mode == "max" else math.inf)
                if self._better(score):
                    self.best_score, bad = score, 0
                    self.best_state = {k: v.detach().cpu().clone()
                                       for k, v in self.model.state_dict().items()}
                else:
                    bad += 1
            self.history.append(rec)
            if verbose:
                extra = "  ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                  if k.startswith("val_") and isinstance(v, float))
                print(f"  [{self.name}] epoch {ep:3d}/{epochs} | loss {rec['train_loss']:.4f} "
                      f"| {extra} | {rec['seconds']:.1f}s")
            if bad >= self.cfg.early_stopping_patience:
                print(f"  [{self.name}] early stop at epoch {ep} "
                      f"(best val_{self.monitor}={self.best_score:.4f})")
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return self.history

    def save(self, path: Path) -> Path:
        torch.save({"state_dict": self.model.state_dict(), "history": self.history,
                    "best_score": self.best_score, "monitor": self.monitor}, path)
        return path


def bce_loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(model(batch), batch["y"])


def ce_loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    return F.cross_entropy(model(batch), batch["y_single"], label_smoothing=0.05)


def contrastive_loss_fn(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    g, t, scale = model(batch)
    loss, _ = info_nce(g, t, scale)
    return loss


def plot_history(history: List[Dict[str, float]], keys: Sequence[str], title: str,
                 fname: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    eps = [h["epoch"] for h in history]
    axes[0].plot(eps, [h["train_loss"] for h in history], marker="o", ms=3, label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("training loss")
    axes[0].legend()
    any_val = False
    for k in keys:
        vals = [h.get(f"val_{k}") for h in history]
        if any(v is not None for v in vals):
            axes[1].plot(eps, vals, marker="o", ms=3, label=f"val {k}")
            any_val = True
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score")
    axes[1].set_title("validation metrics vs. epoch")
    if any_val:
        axes[1].legend()
    fig.suptitle(title, y=1.05)
    p = savefig(fig, fname); plt.show()
    return p
