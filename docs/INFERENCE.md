Inference (Top‑K Recommendations)
================================

Quick Start (Ready‑to‑Run)
--------------------------

To immediately test the model and see predictions, run the ready‑to‑use script in the Feature‑Aware Message Passing folder. It loads the prepared graph and the included checkpoint, computes embeddings, prints top‑k recommendations for a playlist, and reports hits + recall@k for that playlist using the same RandomLinkSplit seed as training. It does not retrain.

From the repo root:

```
python FeatureAwareMessagePassing/inference.py    # defaults: playlist 200, k=300
```

Options:

- `--playlist-id <int>`: choose a specific playlist node (0..num_playlists-1). Defaults to `200`.
- `--k <int>`: top‑k for recommendations and recall (default 300).
- `--data-dir <path>`: where `data_object.pt`, `dataset_stats.json`, `song_info.json`, and `feature_aware_lightgcn_checkpoint.pt` reside (defaults to `FeatureAwareMessagePassing/`).
- `--audio-db <path>`: optional SQLite DB path to include audio features at inference.

What it uses (in the folder):

- `data_object.pt`: PyG graph data.
- `dataset_stats.json`: `{ "num_playlists": ..., "num_nodes": ... }`.
- `song_info.json`: track metadata for pretty printing.
- `feature_aware_lightgcn_checkpoint.pt`: pretrained checkpoint for this variant.

The script reproduces the model architecture used during training (embedding dim 64, 3 layers), loads the checkpoint, builds the normalized message‑passing graph from the train split (to avoid leakage), computes embeddings, and prints recommendations. If `--audio-db` is provided, it loads 7‑dim audio features for songs; otherwise, it uses zeros (fast path).


Manual Recipe (Advanced)
------------------------

If you want to roll your own minimal snippet instead of using the script above, the approach is:

1) Load the graph and dataset stats.
2) Recreate the model with the same hyperparameters used in training (embedding dim, layers, etc.).
3) Run propagation to get node embeddings.
4) Score all songs for a target playlist and take the top‑k, excluding already‑known songs.

See the training scripts for model class definitions and evaluation utilities.
