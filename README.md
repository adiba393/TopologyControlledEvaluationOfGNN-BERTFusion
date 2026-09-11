# GNN–BERT Music Context Understanding

Code for the paper *Topology-Controlled Evaluation of GNN–BERT Fusion for Music Context Understanding under a Measured Resolution Limit* (Tahsin, Faruk, Mostakim — BRAC University).

A four-task system over FMA-small and MusicCaps — BERT tagging, GNN genre recognition, GNN–BERT fusion, and contrastive audio–text retrieval — evaluated under a measured noise floor. The main finding is that message passing over the segment graphs contributes nothing measurable: deleting every edge changes macro-F1 by +0.0100 against a run-to-run resolution limit of 0.0314.

## Installation

```bash
pip install torch numpy pandas scikit-learn matplotlib tqdm
pip install librosa soundfile transformers
pip install torch-geometric   # optional; built-in SAGE/GAT used as fallback
```

Python ≥ 3.10. Experiments ran on a single NVIDIA RTX A6000.

## Data

```
data/raw/
├── fma_small/          FMA-small audio (7,993 tracks, 8 genres)
├── fma_metadata/       tracks.csv, genres.csv
├── musiccaps/          musiccaps-public.csv + audio/
└── DEAM/               valence/arousal annotations
```

Paths are set in `PathConfig` (`config.py`). MusicCaps audio is best taken from the CLAPv2 mirror (96.9% yield). Official splits are used and audited for artist leakage. Set `RunConfig.data_mode = "demo"` to run the whole pipeline on synthetic audio without any corpus on disk.

## Usage

```python
CFG = Config()
set_seed(CFG.train.seed)
DEVICE = pick_device()

table = build_fma_table(CFG)
keys = build_cache(table, cache_dir, CFG, seconds=30.0, win_s=5.0, hop_s=2.5, tag="fma")

model = GNNClassifier(SEG_DIM, CHORD_DIM, len(FMA_GENRES_SMALL), CFG.model)
Trainer(model, ce_loss_fn, CFG.train, monitor="macro_f1", name="task2") \
    .fit(loaders["training"], loaders["validation"], CFG.train.epochs_task2, eval_fn=eval_genre)
```

Fusion ablation (four modes at an equal budget):

```python
for mode in ("xattn", "concat", "bert_only", "gnn_only"):
    run_fusion_variant(mode, CFG.train.epochs_task3, loaders, n_labels, tag=mode)
```

## Structure

| File | Contents |
|---|---|
| `config.py` | All configuration dataclasses |
| `datasets.py` | Corpus tables, label spaces, leakage audit, Dataset and collation |
| `audio_features.py` | Feature extraction and per-track graph cache |
| `graph_builder.py` | Segment and chord graph construction |
| `bert_encoder.py` | Text encoder and tag classifier (Task 1) |
| `gnn_model.py` | SAGE/GAT layers, dual GNN encoder (Task 2), mel-CNN baseline |
| `fusion_model.py` | Cross-attention fusion with ablation harness (Task 3) |
| `contrastive.py` | Dual encoder, InfoNCE, retrieval (Task 4) |
| `train.py` | Trainer, optimiser, schedules, losses |
| `evaluate.py` | Metrics and threshold tuning |

Modules are exported from the master notebook and share its globals (`CFG`, `DEVICE`, `TOKENIZER`, `SEG_DIM`, …), so they are not importable standalone — run the notebook or bind those names first.

## Results

**FMA-small, 8-way genre (macro-F1)**

| Model | Score |
|---|---|
| CLAP linear probe (ref) | 0.5395 |
| Mel-CNN baseline | 0.5245 ± 0.0110 |
| GNN, no edges | 0.4344 ± 0.0196 |
| GNN, full graph | 0.4324 ± 0.0112 |
| Random | 0.1350 |

**MusicCaps, 50-aspect (macro-F1)**

| Model | Score |
|---|---|
| Fusion (graph + caption) | 0.6389 |
| BERT fine-tuned | 0.6375 |
| TF-IDF + logreg | 0.6262 |
| Lexical match (0 params) | 0.5175 |

Resolution limit σ̂<sub>run</sub> = 0.0314 macro-F1, measured over three reruns of one fixed configuration; no smaller difference is reported as a difference. Retrieval reaches R@5 = 0.1421 over the full 802-clip gallery (22.8× chance).

## Reproducing the audits

- **Topology** — retrain under five edge policies (`all`, `q25`, `q50`, `temporal`, `self_only`) holding node features, encoder, readout and schedule fixed.
- **Annotation** — partition MusicCaps test positives by whether the aspect string appears verbatim in the caption; for FMA, retrain under five degraded text views against the audio-only floor.
- **Mechanism** — compute effective rank and cosine profiles per layer via `coherence_analysis()`.

Differences use a paired bootstrap (B = 2000, α = 0.05) with Holm correction.

## Citation

```bibtex
@inproceedings{tahsin_topology_controlled,
  title  = {Topology-Controlled Evaluation of {GNN}--{BERT} Fusion for Music Context
            Understanding under a Measured Resolution Limit},
  author = {Tahsin, Adiba and Faruk, Farhan and Mostakim, Moin},
  year   = {}
}
```

## License

TBD. FMA, MusicCaps and DEAM carry their own terms of use.
