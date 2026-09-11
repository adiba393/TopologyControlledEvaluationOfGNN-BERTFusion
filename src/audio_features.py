"""audio_features.py - part of the GNN-BERT music-context project.
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



class AudioProvider:
    """Uniform access to a waveform, whether it comes from disk or from the synthesiser."""

    def __init__(self, sample_rate: int, seconds: float, seed: int = 0):
        self.sr = int(sample_rate)
        self.seconds = float(seconds)
        self.seed = int(seed)

    def _synth(self, key: Any, attrs: Optional[Dict[str, Any]]) -> np.ndarray:
        h = abs(hash((str(key), self.seed))) % (2 ** 31)
        rng = np.random.default_rng(h)
        if attrs is None:
            attrs = sample_fma_like_attrs(FMA_GENRES_SMALL[h % 8], rng)
        return synth_clip(attrs, self.seconds, self.sr, rng)

    def load(self, key: Any, path: Optional[str], attrs: Optional[Dict[str, Any]] = None,
             offset: float = 0.0) -> Optional[np.ndarray]:
        if path and Path(path).exists():
            try:
                y, _ = librosa.load(path, sr=self.sr, mono=True,
                                    offset=float(offset), duration=self.seconds)
                if y is None or y.size < self.sr:          # < 1 s of audio -> unusable
                    return None
                y = librosa.util.fix_length(y, size=int(self.seconds * self.sr))
                peak = float(np.max(np.abs(y)) + 1e-8)
                return (y / peak).astype(np.float32)
            except Exception:
                return None
        return self._synth(key, attrs)


def extract_frame_features(y: np.ndarray, cfg: AudioConfig) -> Dict[str, np.ndarray]:
    """Frame-level log-mel / chroma / MFCC / spectral descriptors with per-track normalisation."""
    sr, hop, n_fft = cfg.sample_rate, cfg.hop_length, cfg.n_fft
    fmax = cfg.fmax or sr / 2

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                       n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=fmax)
    logmel = librosa.power_to_db(S, ref=np.max).astype(np.float32)          # (128, T)

    if cfg.chroma_type == "cqt":
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop,
                                            n_chroma=cfg.n_chroma, bins_per_octave=36)
    elif cfg.chroma_type == "cens":
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop, n_chroma=cfg.n_chroma)
    else:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                             n_chroma=cfg.n_chroma)
    chroma = chroma.astype(np.float32)                                       # (12, T)

    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=cfg.n_mfcc).astype(np.float32)          # (20, T)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)[None, :]

    T = min(x.shape[-1] for x in (logmel, chroma, mfcc, cent, bw, roll, zcr, rms, onset))
    spectral = np.concatenate([cent[:, :T], bw[:, :T], roll[:, :T],
                               zcr[:, :T], rms[:, :T]], axis=0).astype(np.float32)       # (5, T)

    def _norm(x: np.ndarray) -> np.ndarray:
        if not cfg.per_track_normalise:
            return x
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True) + 1e-6
        return ((x - mu) / sd).astype(np.float32)

    tempo, beats = librosa.beat.beat_track(onset_envelope=onset[0, :T], sr=sr,
                                           hop_length=hop, units="frames")
    tempo = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0

    return {
        "logmel": logmel[:, :T], "logmel_n": _norm(logmel[:, :T]),
        "chroma": chroma[:, :T],
        "mfcc_n": _norm(mfcc[:, :T]),
        "spectral_n": _norm(spectral),
        "onset": onset[0, :T].astype(np.float32),
        "beats": np.asarray(beats, dtype=np.int64),
        "tempo": tempo,
        "n_frames": int(T),
    }


def pool_time(x: np.ndarray, target: int) -> np.ndarray:
    """Adaptive average-pool the time axis of a (F, T) matrix down/up to `target` frames."""
    t = torch.from_numpy(np.ascontiguousarray(x))[None, ...].float()
    return F.adaptive_avg_pool2d(t, (x.shape[0], target))[0].numpy().astype(np.float32)


SEG_FEATURE_LAYOUT = [("mfcc_mean", 20), ("mfcc_std", 20), ("chroma_mean", 12), ("chroma_std", 12),
                      ("spectral_mean", 5), ("spectral_std", 5), ("onset_mean", 1), ("onset_std", 1),
                      ("mel_bands", 16), ("position", 1), ("duration", 1)]


SEG_FEATURE_DIM = sum(d for _, d in SEG_FEATURE_LAYOUT)          # -> 94


def segment_bounds(n_frames: int, frames_per_sec: float, win_s: float, hop_s: float,
                   max_nodes: int) -> List[Tuple[int, int]]:
    w = max(2, int(round(win_s * frames_per_sec)))
    h = max(1, int(round(hop_s * frames_per_sec)))
    bounds = [(s, min(s + w, n_frames)) for s in range(0, max(1, n_frames - w // 2), h)]
    bounds = [(a, b) for a, b in bounds if b - a >= max(2, w // 3)]
    if not bounds:
        bounds = [(0, n_frames)]
    return bounds[:max_nodes]


def segment_node_features(feats: Dict[str, np.ndarray], bounds: List[Tuple[int, int]]) -> np.ndarray:
    mel16 = pool_time(feats["logmel_n"], feats["n_frames"])
    mel16 = np.stack([mel16[i * 8:(i + 1) * 8].mean(axis=0) for i in range(16)], axis=0)  # (16, T)
    n_seg, n_frames = len(bounds), max(1, feats["n_frames"])
    out = np.zeros((n_seg, SEG_FEATURE_DIM), dtype=np.float32)
    for i, (a, b) in enumerate(bounds):
        sl = slice(a, b)
        parts = [
            feats["mfcc_n"][:, sl].mean(1), feats["mfcc_n"][:, sl].std(1),
            feats["chroma"][:, sl].mean(1), feats["chroma"][:, sl].std(1),
            feats["spectral_n"][:, sl].mean(1), feats["spectral_n"][:, sl].std(1),
            np.array([feats["onset"][sl].mean()]), np.array([feats["onset"][sl].std()]),
            mel16[:, sl].mean(1),
            np.array([(a + b) / (2.0 * n_frames)]),
            np.array([(b - a) / float(n_frames)]),
        ]
        out[i] = np.nan_to_num(np.concatenate(parts).astype(np.float32))
    return out


def process_track(key: Any, path: Optional[str], attrs: Optional[Dict[str, Any]],
                  provider: AudioProvider, cfg: Config,
                  win_s: float, hop_s: float, offset: float = 0.0) -> Optional[Dict[str, Any]]:
    y = provider.load(key, path, attrs, offset=offset)
    if y is None:
        return None
    try:
        feats = extract_frame_features(y, cfg.audio)
    except Exception:
        return None
    if feats["n_frames"] < 8:
        return None

    fps = cfg.audio.sample_rate / cfg.audio.hop_length
    bounds = segment_bounds(feats["n_frames"], fps, win_s, hop_s, cfg.graph.max_segment_nodes)
    node_x = segment_node_features(feats, bounds)
    chroma_seg = np.stack([feats["chroma"][:, a:b].mean(1) for a, b in bounds])
    mfcc_seg = np.stack([feats["mfcc_n"][:, a:b].mean(1) for a, b in bounds])

    # --- v7: per-node log-mel patch -----------------------------------------
    # The same `bounds` that produced the 94 statistics also slice the
    # normalised log-mel, so patch i and node i describe exactly the same
    # window. Stored as float16 (~6 KB/node) to keep the cache small.
    seg_mel = None
    if getattr(cfg.model, "use_segment_patch_encoder", False):
        nb_, nf_ = cfg.model.patch_mel_bins, cfg.model.patch_frames
        lm = feats["logmel_n"]                                  # [n_mels, T]
        band = max(1, lm.shape[0] // nb_)
        lm_b = np.stack([lm[j * band:(j + 1) * band].mean(0) for j in range(nb_)], 0)
        patches = np.zeros((len(bounds), nb_, nf_), dtype=np.float32)
        for i, (a, b) in enumerate(bounds):
            seg_slice = lm_b[:, a:b]
            if seg_slice.shape[1] < 2:
                continue
            # resample the window onto a fixed frame grid
            idx = np.linspace(0, seg_slice.shape[1] - 1, nf_)
            lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
            frac = (idx - lo).astype(np.float32)
            patches[i] = seg_slice[:, lo] * (1 - frac) + seg_slice[:, hi] * frac
        seg_mel = torch.from_numpy(patches).half()

    seg = build_segment_graph(node_x, chroma_seg, mfcc_seg, cfg.graph)
    cpath, cunits = estimate_chord_path(feats, cfg.graph)
    chord = build_chord_graph(cpath, cunits, cfg.graph)

    return {
        "key": str(key),
        "seg_x": torch.from_numpy(seg["x"]),
        "seg_mel": seg_mel,                       # [N, patch_mel_bins, patch_frames] or None
        "seg_edge_index": torch.from_numpy(seg["edge_index"]),
        "seg_edge_weight": torch.from_numpy(seg["edge_weight"]),
        "seg_edge_type": torch.from_numpy(seg["edge_type"]),
        "chord_x": torch.from_numpy(chord["x"]),
        "chord_edge_index": torch.from_numpy(chord["edge_index"]),
        "chord_edge_weight": torch.from_numpy(chord["edge_weight"]),
        "chord_ids": torch.from_numpy(chord["chord_ids"]),
        "chord_path": torch.from_numpy(chord["chord_path"]),
        "mel_cnn": torch.from_numpy(pool_time(feats["logmel_n"], cfg.audio.mel_cnn_frames)).half(),
        "tempo": float(feats["tempo"]),
        "n_segments": int(len(bounds)),
    }


def build_cache(table: pd.DataFrame, cache_dir: Path, cfg: Config, seconds: float,
                win_s: float, hop_s: float, tag: str, workers: Optional[int] = None) -> List[Any]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    cache_dir.mkdir(parents=True, exist_ok=True)
    provider = AudioProvider(cfg.audio.sample_rate, seconds, cfg.train.seed)
    todo = [k for k in table.index if not (cache_dir / f"{k}.pt").exists()]
    print(f"[{tag}] cached={len(table) - len(todo)}  to-process={len(todo)}")

    if todo:
        workers = workers or max(4, min(16, (os.cpu_count() or 4)))
        _offset_reset = []

        def _job(k):
            row = table.loc[k]
            offset = 0.0
            if "start_s" in table.columns and pd.notna(row.get("start_s")) and row.get("audio_path"):
                offset = float(row["start_s"])
                # `start_s` is where the 10 s clip sits inside the ORIGINAL YouTube
                # video (30 s, 230 s, 520 s ...). Files from the HuggingFace mirror are
                # ALREADY trimmed to exactly that window, so seeking to 30 s inside a
                # 10 s file returns zero samples, AudioProvider.load() bails on
                # `y.size < sr`, process_track returns None, and the clip is silently
                # dropped from the corpus. Since almost every MusicCaps row has
                # start_s > 0, that would quietly delete nearly the whole dataset.
                # Only honour the offset when the file is long enough to contain it.
                if offset > 0.0:
                    try:
                        import soundfile as sf
                        if sf.info(str(row["audio_path"])).duration < offset + seconds * 0.5:
                            offset = 0.0
                            _offset_reset.append(1)
                    except Exception:
                        offset = 0.0
                        _offset_reset.append(1)
            rec = process_track(k, row.get("audio_path"), row.get("attrs"), provider,
                                cfg, win_s, hop_s, offset=offset)
            if rec is not None:
                torch.save(rec, cache_dir / f"{k}.pt")
                return k
            return None
        # warm up librosa/numba on a single track first: JIT compilation is not thread-safe
        done = int(_job(todo[0]) is not None)
        todo = todo[1:]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_job, k): k for k in todo}
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"{tag}: features+graphs"):
                done += int(f.result() is not None)
        print(f"[{tag}] newly cached {done}/{len(todo) + 1}")
        if _offset_reset:
            print(f"[{tag}] pre-trimmed audio detected -> start_s offset ignored for "
                  f"{len(_offset_reset)} clips (correct for the HuggingFace mirror)")

    return [k for k in table.index if (cache_dir / f"{k}.pt").exists()]


def synth_clip(attrs: Dict[str, Any], seconds: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Render a short musical excerpt from a latent attribute dict."""
    n = int(seconds * sr)
    y = np.zeros(n, dtype=np.float32)
    tempo = float(attrs["tempo"])
    spb = 60.0 / tempo                                   # seconds per beat
    beats_per_chord = int(attrs.get("beats_per_chord", 4))
    prog = PROGRESSIONS[attrs["progression"]]
    root = int(attrs["root"])
    n_harm = int(attrs["n_harm"])
    t_beat = np.arange(0, seconds, spb)

    # --- harmonic layer: one chord every `beats_per_chord` beats ------------
    chord_starts = t_beat[::beats_per_chord]
    for ci, t0 in enumerate(chord_starts):
        deg, quality = prog[ci % len(prog)]
        triad = MAJ_TRIAD if quality == "maj" else MIN_TRIAD
        dur = min(spb * beats_per_chord, seconds - t0)
        if dur <= 0.05:
            break
        i0 = int(t0 * sr); nn = int(dur * sr)
        tt = np.arange(nn, dtype=np.float32) / sr
        env = _adsr(nn, sr, decay=0.3 * dur, sustain=0.55)
        voice = np.zeros(nn, dtype=np.float32)
        for semi in triad:
            midi = 48 + ((root + deg + semi) % 24)
            f0 = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            for h in range(1, n_harm + 1):
                voice += (0.9 / (h ** 1.25)) * np.sin(2 * np.pi * f0 * h * tt + rng.uniform(0, 2 * np.pi))
        y[i0:i0 + nn] += 0.25 * env * voice / max(1.0, len(triad) * 1.5)

    # --- percussive layer ---------------------------------------------------
    perc = float(attrs["percussive"])
    if perc > 0:
        click = rng.standard_normal(int(0.05 * sr)).astype(np.float32)
        click *= np.exp(-np.linspace(0, 12, click.size, dtype=np.float32))
        for bi, tb in enumerate(t_beat):
            i0 = int(tb * sr)
            if i0 + click.size >= n:
                break
            gain = perc * (1.0 if bi % 4 == 0 else 0.55)
            y[i0:i0 + click.size] += gain * click

    # --- broadband texture --------------------------------------------------
    y += float(attrs["noise"]) * rng.standard_normal(n).astype(np.float32)
    peak = float(np.max(np.abs(y)) + 1e-8)
    return (0.95 * y / peak).astype(np.float32)


def sample_fma_like_attrs(genre: str, rng: np.random.Generator) -> Dict[str, Any]:
    (lo, hi), p_min, n_harm, perc, noise, progs = SYNTH_GENRE_PROFILE[genre]
    prog = str(rng.choice(progs))
    if rng.random() < p_min and prog.startswith("I"):
        prog = "i-VI-III-VII"
    return {
        "root": int(rng.integers(0, 12)),
        "tempo": float(rng.uniform(lo, hi)),
        "progression": prog,
        "n_harm": int(max(2, n_harm + rng.integers(-1, 2))),
        "percussive": float(np.clip(perc + rng.normal(0, 0.08), 0.0, 1.0)),
        "noise": float(np.clip(noise + rng.normal(0, 0.02), 0.0, 0.4)),
        "beats_per_chord": int(rng.choice([2, 4, 4, 8])),
    }


def _adsr(n: int, sr: int, attack: float = 0.01, decay: float = 0.25, sustain: float = 0.6) -> np.ndarray:
    a = max(1, int(attack * sr)); d = max(1, int(decay * sr))
    env = np.full(n, sustain, dtype=np.float32)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    end = min(n, a + d)
    env[a:end] = np.linspace(1.0, sustain, end - a, dtype=np.float32)
    rel = max(1, int(0.05 * sr))
    env[-rel:] *= np.linspace(1.0, 0.0, rel, dtype=np.float32)
    return env
