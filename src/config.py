"""config.py - part of the GNN-BERT music-context project.
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



@dataclass
class PathConfig:
    project_root: str = "./gnn-bert-music-context"
    # --- FMA -------------------------------------------------------------
    fma_audio_dir: str = "./gnn-bert-music-context/data/raw/fma_small"
    fma_metadata_dir: str = "./gnn-bert-music-context/data/raw/fma_metadata"
    # --- MusicCaps -------------------------------------------------------
    musiccaps_csv: str = "./gnn-bert-music-context/data/raw/musiccaps/musiccaps-public.csv"
    musiccaps_audio_dir: str = "./gnn-bert-music-context/data/raw/musiccaps/audio"
    # --- optional emotion dataset (DEAM) ---------------------------------
    deam_dir: str = "./gnn-bert-music-context/data/raw/DEAM"


@dataclass
class AudioConfig:
    sample_rate: int = 22050          # brief: "Resample to 22,050 Hz"
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128                 # brief: "log-mel spectrogram (128 bins)"
    n_chroma: int = 12                # brief: "chroma (12 bins)"
    n_mfcc: int = 20
    fmin: float = 20.0
    fmax: Optional[float] = None      # None -> sample_rate / 2
    chroma_type: str = "cqt"          # {'cqt','cens','stft'} - cqt is best for chord estimation
    per_track_normalise: bool = True  # brief: "normalize per track"
    mel_cnn_frames: int = 512         # fixed time resolution of the cached CNN input
    fma_clip_seconds: float = 30.0
    musiccaps_clip_seconds: float = 10.0


@dataclass
class GraphConfig:
    # brief: "Split each track into fixed windows (e.g., 5-10 s)"
    fma_window_seconds: float = 5.0
    fma_hop_seconds: float = 2.5
    # 10 s MusicCaps clips cannot host 5 s windows and still form a graph -> scaled proportionally
    mc_window_seconds: float = 2.0
    mc_hop_seconds: float = 1.0
    beat_synchronous_chords: bool = True
    similarity_threshold: float = 0.80   # tau in "cosine similarity of MFCC/chroma > tau"
    similarity_topk: int = 3             # cap to keep graphs sparse and comparable
    add_self_loops: bool = True
    chord_median_filter: int = 5         # smoothing of the per-beat chord path
    max_segment_nodes: int = 64
    min_chord_nodes: int = 2


@dataclass
class TextConfig:
    bert_name: str = "distilbert-base-uncased"   # swap to 'bert-base-uncased' for the full model
    max_length_caption: int = 192                # brief: "max length 128-256"
    max_length_metadata: int = 128
    freeze_bert_layers: int = 0                  # 0 = fully fine-tuned; >0 freezes the bottom N blocks
    strip_label_terms_from_metadata: bool = True # prevents trivial genre leakage through FMA free-text


@dataclass
class LabelConfig:
    musiccaps_top_tags: int = 50        # brief Task 1: "top-50 tags"
    fma_max_subgenres: int = 25
    min_positive_count: int = 20        # drop labels that are too rare to score reliably
    use_mood_lexicon: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size_text: int = 32
    batch_size_graph: int = 64
    batch_size_fusion: int = 24
    batch_size_contrastive: int = 48
    epochs_task1: int = 6
    epochs_task2: int = 60
    epochs_task3: int = 12
    epochs_task4: int = 20
    epochs_cnn: int = 40
    lr_bert: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    num_workers: int = 0                # 0 is the safe default inside notebooks on every OS
    amp: bool = True
    # multi-task weights of L = L_tags + alpha*||v-v_hat||^2 + beta*||a-a_hat||^2
    alpha_valence: float = 0.5
    beta_arousal: float = 0.5
    contrastive_temperature: float = 0.07


@dataclass
class ModelConfig:
    gnn_type: str = "sage"        # {'sage','gat'}
    gnn_hidden: int = 256
    gnn_layers: int = 3
    gnn_heads: int = 4            # GAT only
    gnn_dropout: float = 0.3
    graph_embed_dim: int = 256
    fusion_dim: int = 256
    fusion_dropout: float = 0.2
    projection_dim: int = 256     # contrastive space
    cnn_channels: Tuple[int, ...] = (32, 64, 128, 256)


@dataclass
class RunConfig:
    data_mode: str = "auto"       # {'auto','real','demo'}
    n_synth_fma_tracks: int = 480
    n_synth_musiccaps: int = 900
    max_fma_tracks: Optional[int] = None       # e.g. 2000 for a quick real-data pass
    max_musiccaps_clips: Optional[int] = None
    enforce_artist_disjoint_splits: bool = False
    run_task4_zero_shot: bool = True
    n_export_graph_samples: int = 24           # brief: ">= 20 example .pt / .json graphs"


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    text: TextConfig = field(default_factory=TextConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str))
