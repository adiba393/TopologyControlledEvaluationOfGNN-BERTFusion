# GNN-Based BERT for Understanding Context from Music

Course project - Neural Networks (CSE425 / EEE474 / CSE715).
Hybrid **BERT + Graph Neural Network** system for *understanding* musical context: multi-label tagging,
multi-context fusion, and cross-modal retrieval on **FMA-small** and **MusicCaps**.

## Results (REAL DATA)

| Model | Dataset | Macro-F1 | AUC-PR | R@5 |
|---|---|---|---|---|
| B1 random tags | MusicCaps | 0.0945 | 0.0612 | - |
| B2 CNN mel-spectrogram | FMA-small | 0.5386 | 0.5516 | - |
| B4 PCA + MLP | FMA-small | 0.3381 | 0.3246 | - |
| **Task 1** BERT-only (B3) | MusicCaps | 0.6375 | 0.6932 | - |
| **Task 2** GNN-only | FMA-small | 0.4535 | 0.4895 | - |
| **Task 3** GNN-BERT fusion | FMA-small | 0.2929 | 0.3304 | - |
| **Task 4** Contrastive dual-encoder | MusicCaps | 0.0961 | 0.0691 | 0.1334 |

Full numbers, ablations and per-tag breakdowns: `results/metrics.json` and `results/*.csv`.

## Quick start

```bash
pip install -r requirements.txt      # + ffmpeg on PATH
# optional: real data
curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
unzip fma_metadata.zip -d data/raw/ && unzip fma_small.zip -d data/raw/
curl -o data/raw/musiccaps/musiccaps-public.csv \
     https://storage.googleapis.com/gresearch/musiccaps/musiccaps-public.csv
jupyter lab GNN_BERT_Music_Context_Project.ipynb   # run all cells
```

Without the corpora the notebook runs on a deterministic synthetic stand-in so every code path still executes.

## Method

1. **Preprocessing** - 22,050 Hz mono; 128-bin log-mel and 12-bin chroma, per-track standardised; MFCC-20 and five
   spectral descriptors; 5 s windows with 2.5 s hop (2 s / 1 s for 10 s MusicCaps clips); beat tracking for
   beat-synchronous chroma.
2. **Graphs** - (a) *segment graph*: nodes = windows with 94-D descriptors, edges = temporal adjacency plus
   MFCC/chroma cosine similarity above tau=0.8; (b) *chord-transition graph*: beat-synchronous chroma
   matched to 24 major/minor triad templates, nodes = unique chords, edges weighted by observed transition counts.
3. **Models** - dual-branch GraphSAGE/GAT encoder with mean+max readout; BERT (`bert-base-uncased`) text tower;
   cross-attention fusion z = CONCAT(g, A H_text); contrastive dual-encoder with learnable temperature.
4. **Protocol** - official FMA `set/split` partition with an artist-leakage audit; MusicCaps split on the AudioSet-eval
   flag; per-tag decision thresholds tuned on validation only; every experiment seeded at 42.

## Layout

```
gnn-bert-music-context/
  GNN_BERT_Music_Context_Project.ipynb   # master notebook (all four tasks)
  README.md  requirements.txt  config.yaml
  data/{raw,processed,splits}            # graph cache + >= 20 exported sample graphs
  notebooks/{eda,demo_context}.ipynb
  src/{audio_features,graph_builder,datasets,bert_encoder,gnn_model,
       fusion_model,contrastive,train,evaluate,config}.py
  results/{metrics.json,*.csv,plots/,retrieval_examples/,checkpoints/}
  report/final_report.pdf
```

## Report templates
NeurIPS 2024 - <https://www.overleaf.com/latex/templates/neurips-2024/tpsbbrdqcmsh> |
IEEE Conference - <https://www.overleaf.com/latex/templates/ieee-conference-template> |
ICML 2025 - <https://www.overleaf.com/latex/templates/icml2025-template>

## Data credits
FMA: Defferrard, Benzi, Vandergheynst & Bresson, *FMA: A Dataset For Music Analysis*, ISMIR 2017.
MusicCaps: Agostinelli et al., *MusicLM: Generating Music From Text*, 2023 (captions CC-BY-SA; audio not redistributed).
