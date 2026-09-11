"""datasets.py - part of the GNN-BERT music-context project.
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



class LabelSpace:
    """Ordered multi-label vocabulary with frequency-filtered construction."""

    def __init__(self, names: Sequence[str], groups: Optional[Dict[str, str]] = None):
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.groups = groups or {n: "tag" for n in self.names}

    def __len__(self) -> int:
        return len(self.names)

    def encode(self, tags: Iterable[str]) -> np.ndarray:
        v = np.zeros(len(self.names), dtype=np.float32)
        for t in tags:
            i = self.index.get(str(t).strip().lower())
            if i is not None:
                v[i] = 1.0
        return v

    def decode(self, vec: np.ndarray, thresh: float = 0.5) -> List[str]:
        return [n for n, p in zip(self.names, np.asarray(vec).ravel()) if p >= thresh]

    def top_k(self, probs: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        order = np.argsort(-np.asarray(probs).ravel())[:k]
        return [(self.names[i], float(probs[i])) for i in order]


def build_musiccaps_label_space(table: pd.DataFrame, top_k: int, train_mask: np.ndarray,
                                min_count: int) -> LabelSpace:
    cnt = Counter()
    for asp in table.loc[train_mask, "aspect_list"]:
        cnt.update({str(a).strip().lower() for a in asp if str(a).strip()})
    names = [a for a, c in cnt.most_common() if c >= min_count][:top_k]
    if len(names) < 5:                                   # tiny corpora -> relax the frequency floor
        names = [a for a, _ in cnt.most_common(top_k)]
    return LabelSpace(names, {n: "aspect" for n in names})


def build_fma_label_space(table: pd.DataFrame, train_mask: np.ndarray, cfg: LabelConfig) -> LabelSpace:
    names, groups = [], {}
    for g in FMA_GENRES_SMALL:                            # 1) the 8 top-level genres
        names.append(g.lower()); groups[g.lower()] = "genre_top"

    sub = Counter()                                       # 2) frequent sub-genres from genres_all
    for gl in table.loc[train_mask, "genres_all"]:
        for g in gl:
            gl_ = str(g).strip().lower()
            if gl_ and gl_ not in groups:
                sub[gl_] += 1
    for g, c in sub.most_common(cfg.fma_max_subgenres):
        if c >= cfg.min_positive_count:
            names.append(g); groups[g] = "genre_sub"

    if cfg.use_mood_lexicon:                              # 3) mood tags from the free-text tag fields
        mood = Counter()
        for tl in table.loc[train_mask, "tags"]:
            joined = " ".join(str(t).lower() for t in tl)
            for m in MOOD_LEXICON:
                if m in joined:
                    mood[m] += 1
        for m, c in mood.most_common():
            if c >= cfg.min_positive_count and m not in groups:
                names.append(m); groups[m] = "mood"
    return LabelSpace(names, groups)


def fma_row_tags(row: pd.Series, space: LabelSpace) -> List[str]:
    out = {str(row["genre_top"]).lower()}
    out.update(str(g).strip().lower() for g in row["genres_all"])
    joined = " ".join(str(t).lower() for t in row["tags"])
    out.update(m for m in MOOD_LEXICON if m in joined)
    return [t for t in out if t in space.index]


def strip_terms(text: str, terms: Sequence[str]) -> str:
    if not terms:
        return text
    pat = re.compile(r"\b(" + "|".join(sorted((re.escape(t) for t in terms), key=len, reverse=True)) + r")\b",
                     flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", pat.sub(" ", text)).strip()


def fma_text_view(row: pd.Series, banned: Sequence[str]) -> str:
    tags = ", ".join(sorted({str(t).strip().lower() for t in row["tags"] if str(t).strip()})[:12])
    txt = (f"Track: {row['track_title']}. Album: {row['album_title']}. "
           f"Artist: {row['artist_name']}. Tags: {tags if tags else 'none'}.")
    return strip_terms(txt, banned)


def audit_artist_leakage(table: pd.DataFrame) -> Dict[str, Any]:
    by = {s: set(g["artist_id"]) for s, g in table.groupby("split")}
    tr, va, te = by.get("training", set()), by.get("validation", set()), by.get("test", set())
    return {"n_artists_train": len(tr), "n_artists_val": len(va), "n_artists_test": len(te),
            "overlap_train_test": len(tr & te), "overlap_train_val": len(tr & va),
            "leak_free": len(tr & te) == 0 and len(tr & va) == 0}


def enforce_artist_disjoint(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = table.copy()
    for aid, grp in table.groupby("artist_id"):
        if grp["split"].nunique() > 1:
            majority = grp["split"].value_counts().idxmax()
            out.loc[grp.index, "split"] = majority
    for s in ("validation", "test"):                      # never let a split become empty
        if (out["split"] == s).sum() == 0:
            pool = out.index[out["split"] == "training"].to_numpy()
            take = rng.choice(pool, size=max(1, len(pool) // 10), replace=False)
            out.loc[take, "split"] = s
    return out


@dataclass
class GraphBatch:
    x: torch.Tensor                      # [sum_N, F]  hand-crafted node stats
    edge_index: torch.Tensor             # [2, sum_E]
    edge_weight: torch.Tensor            # [sum_E]
    batch: torch.Tensor                  # [sum_N] graph id per node
    num_graphs: int
    mel: Optional[torch.Tensor] = None   # [sum_N, bins, frames] v7 node patches

    def to(self, device, **kwargs) -> "GraphBatch":
        return GraphBatch(self.x.to(device, non_blocking=True),
                          self.edge_index.to(device, non_blocking=True),
                          self.edge_weight.to(device, non_blocking=True),
                          self.batch.to(device, non_blocking=True), self.num_graphs,
                          None if self.mel is None else self.mel.to(device, non_blocking=True))


def collate_graphs(xs: List[torch.Tensor], eis: List[torch.Tensor],
                   ews: List[torch.Tensor],
                   mels: Optional[List[Optional[torch.Tensor]]] = None) -> GraphBatch:
    offset, X, EI, EW, B = 0, [], [], [], []
    for gi, (x, ei, ew) in enumerate(zip(xs, eis, ews)):
        n = x.shape[0]
        X.append(x); EI.append(ei + offset); EW.append(ew)
        B.append(torch.full((n,), gi, dtype=torch.long))
        offset += n
    M = None
    if mels is not None and all(m is not None for m in mels):
        M = torch.cat([m.float() for m in mels], 0)
    return GraphBatch(torch.cat(X, 0).float(), torch.cat(EI, 1).long(),
                      torch.cat(EW, 0).float(), torch.cat(B, 0), len(xs), M)


class MusicDataset(Dataset):
    """Serves any subset of {graph, mel, text} plus targets for one split of one corpus."""

    def __init__(self, keys: Sequence[Any], cache_dir: Path, texts: Sequence[str],
                 y_multi: Optional[np.ndarray] = None, y_single: Optional[np.ndarray] = None,
                 tokenizer=None, max_length: int = 128,
                 modalities: Sequence[str] = ("graph", "mel", "text"),
                 feature_norm: Optional[Dict[str, torch.Tensor]] = None):
        self.keys = list(keys)
        self.cache_dir = Path(cache_dir)
        self.texts = list(texts)
        self.y_multi = y_multi
        self.y_single = y_single
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.modalities = set(modalities)
        self.fn = feature_norm
        if "text" in self.modalities and tokenizer is not None:
            enc = tokenizer(self.texts, padding="max_length", truncation=True,
                            max_length=max_length, return_tensors="pt")
            self.input_ids, self.attn = enc["input_ids"], enc["attention_mask"]

    def __len__(self) -> int:
        return len(self.keys)

    def _norm(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        if self.fn is None:
            return x
        return (x - self.fn[f"{kind}_mean"]) / self.fn[f"{kind}_std"]

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {"idx": i}
        if {"graph", "mel"} & self.modalities:
            rec = safe_torch_load(self.cache_dir / f"{self.keys[i]}.pt")
            if "graph" in self.modalities:
                item["seg_x"] = self._norm(rec["seg_x"], "seg")
                item["seg_mel"] = rec.get("seg_mel")
                item["seg_ei"] = rec["seg_edge_index"]
                item["seg_ew"] = rec["seg_edge_weight"]
                item["chord_x"] = self._norm(rec["chord_x"], "chord")
                item["chord_ei"] = rec["chord_edge_index"]
                item["chord_ew"] = rec["chord_edge_weight"]
            if "mel" in self.modalities:
                item["mel"] = rec["mel_cnn"].float()
        if "text" in self.modalities and self.tokenizer is not None:
            item["input_ids"] = self.input_ids[i]
            item["attention_mask"] = self.attn[i]
        if self.y_multi is not None:
            item["y"] = torch.from_numpy(self.y_multi[i])
        if self.y_single is not None:
            item["y_single"] = torch.tensor(int(self.y_single[i]))
        return item


def make_collate(modalities: Sequence[str]) -> Callable:
    mods = set(modalities)

    def _collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"idx": torch.tensor([it["idx"] for it in items])}
        if "graph" in mods:
            out["seg"] = collate_graphs([it["seg_x"] for it in items],
                                        [it["seg_ei"] for it in items],
                                        [it["seg_ew"] for it in items],
                                        [it.get("seg_mel") for it in items])
            out["chord"] = collate_graphs([it["chord_x"] for it in items],
                                          [it["chord_ei"] for it in items],
                                          [it["chord_ew"] for it in items])
        if "mel" in mods:
            out["mel"] = torch.stack([it["mel"] for it in items]).unsqueeze(1)
        if "text" in mods and "input_ids" in items[0]:
            out["input_ids"] = torch.stack([it["input_ids"] for it in items])
            out["attention_mask"] = torch.stack([it["attention_mask"] for it in items])
        if "y" in items[0]:
            out["y"] = torch.stack([it["y"] for it in items])
        if "y_single" in items[0]:
            out["y_single"] = torch.stack([it["y_single"] for it in items])
        return out

    return _collate


def move_batch(batch: Dict[str, Any], device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, (torch.Tensor, GraphBatch)) else v
    return out


def fit_feature_norm(keys: Sequence[Any], cache_dir: Path, max_tracks: int = 800) -> Dict[str, torch.Tensor]:
    seg, chord = [], []
    for k in list(keys)[:max_tracks]:
        r = safe_torch_load(Path(cache_dir) / f"{k}.pt")
        seg.append(r["seg_x"]); chord.append(r["chord_x"])
    S, C = torch.cat(seg, 0), torch.cat(chord, 0)
    return {"seg_mean": S.mean(0), "seg_std": S.std(0) + 1e-5,
            "chord_mean": C.mean(0), "chord_std": C.std(0) + 1e-5}


def split_indices(table: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {s: np.where((table["split"] == s).to_numpy())[0]
            for s in ("training", "validation", "test")}


def load_fma_metadata(metadata_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Faithful re-implementation of the official `fma/utils.py::load` for tracks.csv / genres.csv."""
    md = Path(metadata_dir)
    tracks = pd.read_csv(md / "tracks.csv", index_col=0, header=[0, 1])
    for col in [("track", "tags"), ("album", "tags"), ("artist", "tags"),
                ("track", "genres"), ("track", "genres_all")]:
        if col in tracks.columns:
            tracks[col] = tracks[col].map(lambda v: ast.literal_eval(v) if isinstance(v, str) else [])
    for col in [("track", "genre_top"), ("album", "type"), ("set", "subset"), ("set", "split")]:
        if col in tracks.columns:
            tracks[col] = tracks[col].astype("category")
    tracks = tracks[pd.to_numeric(tracks.index, errors="coerce").notna()]
    tracks.index = tracks.index.astype(int)
    genres = pd.read_csv(md / "genres.csv", index_col=0)
    return tracks, genres


def fma_audio_path(audio_dir: str, track_id: int) -> Path:
    tid = f"{int(track_id):06d}"
    return Path(audio_dir) / tid[:3] / f"{tid}.mp3"


def build_fma_table(cfg: Config) -> pd.DataFrame:
    """Normalise FMA-small into the single flat table the rest of the notebook uses."""
    tracks, genres = load_fma_metadata(cfg.paths.fma_metadata_dir)
    small = tracks[tracks[("set", "subset")] == "small"].copy()
    gid2name = genres["title"].to_dict()

    rows = []
    for tid, r in small.iterrows():
        if int(tid) in FMA_CORRUPT_IDS:
            continue
        p = fma_audio_path(cfg.paths.fma_audio_dir, tid)
        if not p.exists():
            continue
        gall = [gid2name.get(int(g), str(g)) for g in (r[("track", "genres_all")] or [])]
        tags = []
        for scope in ("track", "album", "artist"):
            v = r.get((scope, "tags"), [])
            if isinstance(v, (list, tuple)):
                tags.extend([str(t).strip().lower() for t in v if str(t).strip()])
        rows.append({
            "track_id": int(tid),
            "genre_top": str(r[("track", "genre_top")]),
            "genres_all": gall,
            "artist_id": int(r[("artist", "id")]) if not pd.isna(r.get(("artist", "id"))) else -1,
            "artist_name": str(r.get(("artist", "name"), "")),
            "track_title": str(r.get(("track", "title"), "")),
            "album_title": str(r.get(("album", "title"), "")),
            "tags": sorted(set(tags)),
            "split": str(r[("set", "split")]),
            "audio_path": str(p),
            "attrs": None,
        })
    df = pd.DataFrame(rows).set_index("track_id")
    df = df[df["genre_top"].isin(FMA_GENRES_SMALL)]
    if cfg.run.max_fma_tracks:
        keep = (df.groupby("genre_top", group_keys=False)
                  .apply(lambda g: g.head(max(1, cfg.run.max_fma_tracks // len(FMA_GENRES_SMALL)))))
        df = keep
    return df


def build_musiccaps_table(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.paths.musiccaps_csv)
    df = df.drop_duplicates(subset="ytid").set_index("ytid")
    df["aspect_list"] = df["aspect_list"].map(
        lambda v: [str(a).strip().lower() for a in ast.literal_eval(v)] if isinstance(v, str) else [])
    if "is_audioset_eval" not in df.columns:
        df["is_audioset_eval"] = False
    df["is_audioset_eval"] = df["is_audioset_eval"].astype(bool)
    adir = Path(cfg.paths.musiccaps_audio_dir)
    def _find(y):
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg"):
            p = adir / f"{y}{ext}"
            if p.exists():
                return str(p)
        return None
    df["audio_path"] = [_find(y) for y in df.index]
    df["attrs"] = None
    if cfg.run.max_musiccaps_clips:
        df = df.head(cfg.run.max_musiccaps_clips)
    return df[["start_s", "end_s", "caption", "aspect_list", "is_audioset_eval", "audio_path", "attrs"]]


def build_synthetic_fma(n_tracks: int, seed: int) -> pd.DataFrame:
    """A DataFrame with exactly the FMA columns the rest of the notebook consumes."""
    rng = np.random.default_rng(seed)
    styles = ["Analog", "Midnight", "Golden", "Paper", "Neon", "River", "Iron", "Velvet",
              "Silent", "Crimson", "Glass", "Northern"]
    nouns = ["Machine", "Highway", "Letters", "Cathedral", "Static", "Bloom", "Signal",
             "Harbour", "Fever", "Lantern", "Circuit", "Ashes"]
    # Artists have a home genre (as in the real corpus) so that splits can be BOTH
    # genre-stratified AND artist-disjoint -- the brief forbids artist leakage.
    n_artists = len(FMA_GENRES_SMALL) * max(4, n_tracks // (8 * len(FMA_GENRES_SMALL)))
    artist_genre = {a: FMA_GENRES_SMALL[a % len(FMA_GENRES_SMALL)] for a in range(n_artists)}
    rows = []
    for i in range(n_tracks):
        aid = i % n_artists
        genre = artist_genre[aid]
        mood = str(rng.choice(MOOD_LEXICON))
        tags = [mood, str(rng.choice(["lofi", "live", "acoustic", "remix", "instrumental", "demo"]))]
        rows.append({
            "track_id": 100000 + i,
            "genre_top": genre,
            "genres_all": [genre] + ([f"{genre} Sub-{int(rng.integers(1, 4))}"] if rng.random() < 0.6 else []),
            "artist_id": aid,
            "artist_name": f"{styles[aid % len(styles)]} {nouns[(aid * 7) % len(nouns)]}",
            "track_title": f"{rng.choice(styles)} {rng.choice(nouns)}",
            "album_title": f"{rng.choice(nouns)} Sessions",
            "tags": tags,
            "split": "training",           # replaced by the grouped splitter below
            "attrs": sample_fma_like_attrs(genre, rng),
        })
    df = pd.DataFrame(rows).set_index("track_id")

    # Genre-stratified 80/10/10 split assigned at ARTIST granularity: every track by a
    # given artist lands in exactly one split, so the audit in Section 4 comes back clean.
    for genre, grp in df.groupby("genre_top"):
        artists = np.array(sorted(grp["artist_id"].unique())); rng.shuffle(artists)
        n_hold = max(1, int(round(0.1 * len(artists))))
        val_a = set(artists[:n_hold].tolist())
        test_a = set(artists[n_hold:2 * n_hold].tolist())
        for split_name, members in (("validation", val_a), ("test", test_a)):
            sel = grp.index[grp["artist_id"].isin(members)]
            df.loc[sel, "split"] = split_name
    return df


def build_synthetic_musiccaps(n_clips: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows = []
    for i in range(n_clips):
        pick = {k: str(rng.choice(v)) for k, v in MC_SYNTH_VOCAB.items()}
        aspects = list(pick.values())
        caption = (f"This is a {pick['tempo']} {pick['genre']} piece featuring {pick['instrument']}. "
                   f"The mood is {pick['mood']} and the track is {pick['vocal']}. "
                   f"It sounds like a {pick['production']} with a steady rhythmic pulse.")
        attrs = sample_fma_like_attrs(MC_GENRE_TO_FMA[pick["genre"]], rng)
        attrs["tempo"] = {"slow tempo": 68.0, "medium tempo": 104.0, "fast tempo": 142.0}[pick["tempo"]]
        if pick["mood"] in {"dark", "melancholic", "aggressive", "nostalgic"}:
            attrs["progression"] = "i-VI-III-VII"
        rows.append({"ytid": f"SYN{i:05d}", "start_s": 0, "end_s": 10,
                     "caption": caption, "aspect_list": aspects,
                     "is_audioset_eval": bool(i % 10 == 0), "attrs": attrs})
    return pd.DataFrame(rows).set_index("ytid")


def download_musiccaps_audio(csv_path: str, out_dir: str, limit: Optional[int] = None,
                             sample_rate: int = 22050,
                             workers: int = 6, seed: int = 42,
                             eval_share: float = 0.35) -> Dict[str, Any]:
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ff_bin = locate_ffmpeg()
    if ff_bin is None:
        raise RuntimeError(
            "ffmpeg not found. yt-dlp needs it to trim and convert to wav.\n"
            "  Option 1 (no solver risk):  pip install imageio-ffmpeg\n"
            "  Option 2:                   conda install -c conda-forge ffmpeg")
    print(f"ffmpeg binary : {ff_bin}")
    print(f"ffmpeg version: {verify_ffmpeg(ff_bin)}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)

    # Stratified subsample. Without a limit we take everything, so the ordering
    # is irrelevant; with a limit we keep enough eval clips to fill the test
    # split without letting them crowd out the training data.
    if limit and "is_audioset_eval" in df.columns:
        ev = df[df["is_audioset_eval"].astype(bool)]
        rest = df[~df["is_audioset_eval"].astype(bool)]
        n_ev = min(len(ev), int(round(limit * eval_share)))
        n_rest = min(len(rest), limit - n_ev)
        n_ev = min(len(ev), limit - n_rest)          # give leftovers back to eval
        df = pd.concat([ev.sample(n=n_ev, random_state=seed),
                        rest.sample(n=n_rest, random_state=seed)], ignore_index=True)
        print(f"stratified sample: {n_ev} eval + {n_rest} non-eval = {len(df)} clips")
    elif limit:
        df = df.sample(n=min(limit, len(df)), random_state=seed)

    rows = list(df.itertuples())
    already = sum(1 for r in rows if (out / f"{r.ytid}.wav").exists())
    todo = [r for r in rows if not (out / f"{r.ytid}.wav").exists()]
    print(f"musiccaps: {already} already on disk, {len(todo)} to fetch "
          f"({workers} parallel workers, {MUSICCAPS_SLEEP_REQUESTS}s between requests)")

    # ---- PREFLIGHT: a few clips, errors visible, abort the batch if all fail --
    if todo:
        preflight(todo, out, ff_bin, sample_rate, n=4)
        todo = [r for r in todo if not (out / f"{r.ytid}.wav").exists()]

    fail = []

    def _one(row):
        dst = out / f"{row.ytid}.wav"
        if dst.exists():
            return True
        cmd = ytdlp_cmd(row, out, ff_bin, sample_rate) + ["-q"]
        try:
            subprocess.run(cmd, check=True, timeout=180,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return dst.exists()
        except Exception as exc:
            fail.append((row.ytid, type(exc).__name__))
            return False

    ok = sum(1 for r in rows if (out / f"{r.ytid}.wav").exists())
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_one, r) for r in todo]
            for f in tqdm(as_completed(futs), total=len(futs), desc="musiccaps/yt-dlp"):
                ok += int(bool(f.result()))

    n_eval = 0
    if "is_audioset_eval" in df.columns:
        n_eval = int(sum((out / f"{r.ytid}.wav").exists()
                         for r in rows if bool(getattr(r, "is_audioset_eval", False))))
    return {"requested": len(rows), "downloaded": ok, "failed": len(fail),
            "eval_clips_on_disk": n_eval,
            "yield": round(ok / max(1, len(rows)), 3)}
