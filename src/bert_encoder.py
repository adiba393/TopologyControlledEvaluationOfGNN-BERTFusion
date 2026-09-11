"""bert_encoder.py - part of the GNN-BERT music-context project.
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



class BertTextEncoder(nn.Module):
    """Wraps a HuggingFace encoder; returns the full token sequence H_text and the CLS vector t."""

    def __init__(self, name: str, freeze_bottom: int = 0):
        super().__init__()
        # `eager` attention is REQUIRED: the fused/sdpa kernels silently return no attention
        # maps, which would break the Task 1 / Task 3 attention visualisations.
        try:
            self.bert = AutoModel.from_pretrained(name, attn_implementation="eager")
        except (TypeError, ValueError):
            self.bert = AutoModel.from_pretrained(name)
        conf = self.bert.config
        self.out_dim = int(getattr(conf, "hidden_size", 0) or getattr(conf, "dim"))
        if freeze_bottom > 0:
            for prm in self.bert.embeddings.parameters():
                prm.requires_grad = False
            layers = []
            for attr, sub in (("encoder", "layer"), ("transformer", "layer")):
                block = getattr(self.bert, attr, None)
                if block is not None and hasattr(block, sub):
                    layers = list(getattr(block, sub)); break
            for layer in layers[:freeze_bottom]:
                for prm in layer.parameters():
                    prm.requires_grad = False

    def forward(self, input_ids, attention_mask, output_attentions: bool = False):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        output_attentions=output_attentions)
        H = out.last_hidden_state                     # [B, L, d]
        return (H, H[:, 0], out.attentions) if output_attentions else (H, H[:, 0])


class BertTagClassifier(nn.Module):
    """Task 1:  t = BERT_CLS(X_text);  y_hat_k = sigmoid(w_k^T t + b_k)."""

    def __init__(self, name: str, n_labels: int, dropout: float = 0.2, freeze_bottom: int = 0):
        super().__init__()
        self.encoder = BertTextEncoder(name, freeze_bottom)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.encoder.out_dim, n_labels))

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        _, t = self.encoder(batch["input_ids"], batch["attention_mask"])
        return self.head(t)
