Training & Evaluation
=====================

This page explains how to run each LightGCN variant, what data each expects, and what you should see during training.


Common Inputs
-------------

Each variant expects these files in its folder:

- `data_object.pt` — PyTorch Geometric `Data` graph object.
- `dataset_stats.json` — contains `num_playlists` and `num_nodes`.
- `playlist_info.json`, `song_info.json` — optional metadata used for analysis/feature mapping.

Feature-aware variants additionally require:

- `spotify_mpd_audio_features.db` — SQLite database containing audio feature columns (e.g., `danceability`, `energy`, `tempo`, ...).

Note: We do not rely on `hyperparameters.txt` in the project root; hyperparameters are defined in the scripts below.


Base Setup
----------

Folder: `BaseSetup/`

Run:

- `python BaseSetup/base_model_lightgcn.py`

Highlights:

- Implements plain LightGCN with BPR loss and recall@k evaluation.
- Uses `RandomLinkSplit` to form train/val/test splits.
- Saves embeddings during validation when `save_emb_dir` is set (default `embeddings/`).

Tuning:

- Edit the hyperparameters defined near the bottom of the script (epochs, k, num_layers, batch_size, embedding_dim).
- GPU is picked automatically if available.


Performance-Optimized
---------------------

Folder: `BaseSetupWithPerfOptimisations/`

Run:

- `python BaseSetupWithPerfOptimisations/base_model_lightgcn_with_perf_optimisations.py`

Highlights:

- Adds mixed precision (AMP) and small efficiency tweaks for faster training.
- Keeps model semantics aligned with the base version.

Tuning:

- Hyperparameters are similarly defined near the bottom of the script.
- AMP is used automatically when supported by your torch install/GPU.


Feature-Aware Initialization
----------------------------

Folder: `FeatureAwareInitialization/`

Run:

- `python FeatureAwareInitialization/base_model_lightgcn_with_feature_aware_embedding_Initialization.py`

Highlights:

- Loads audio features from `spotify_mpd_audio_features.db` and uses them to initialize node embeddings.
- Retains the LightGCN message passing and training loop.

Tuning:

- Ensure the SQLite DB path and expected feature columns match your database schema.
- Hyperparameters (epochs, k, layers, embedding_dim, etc.) are set in the script.


Feature-Aware Message Passing
-----------------------------

Folder: `FeatureAwareMessagePassing/`

Run:

- `python FeatureAwareMessagePassing/base_model_lightgcn_with_feature_aware_embedding_Initialization_and_feature_aware_message_passing.py`

Highlights:

- Incorporates audio features both in initialization and in the message passing step.
- Uses similar evaluation/logging patterns as the other variants.

Tuning:

- Same as above: verify DB availability and edit hyperparameters in the script.


Expected Output
---------------

All scripts print:

- Training loss per epoch.
- Periodic validation recall@k with timing/throughput info.
- Final test recall@k.

If `save_emb_dir` is set (base script), per-epoch embeddings will be saved into that directory, which can be used for analysis/visualization.
