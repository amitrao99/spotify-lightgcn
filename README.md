Spotify LightGCN Recommender
=================================

LightGCN-based recommender systems for the Spotify Million Playlist Dataset, packaged as four incremental variants:

- Base Setup: clean LightGCN implementation and training loop.
- Performance-Optimized: faster training with mixed precision and small efficiency improvements.
- Feature-Aware Initialization: uses audio features (from a SQLite DB) to initialize embeddings.
- Feature-Aware Message Passing: extends the model to pass feature-derived messages.

This repository is thesis-backed and structured for clarity and reproducibility.


Quick Start
-----------

1) Create an environment (Python 3.10 or 3.11 recommended):

   - venv: `python -m venv .venv && .\\.venv\\Scripts\\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)
   - Conda: `conda create -n lightgcn python=3.10 -y && conda activate lightgcn`

2) Install PyTorch (CPU or GPU) following https://pytorch.org/get-started/locally/. Example (CUDA 12.1):

   `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`

   Or CPU-only: `pip install torch torchvision torchaudio`

3) Install PyTorch Geometric following the official matrix for your torch/CUDA versions: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

   Typical (example; adjust versions to your torch):
   - `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html`
   - `pip install torch_geometric`

4) Install project requirements:

   - Simple: `pip install -r requirements.txt`
   - Guided (torch + PyG + deps): `python requirements.py --dry-run` to preview, then e.g. `python requirements.py --cuda cu121 --yes` or `python requirements.py --cpu --yes`

5) Run the base model (expects prepared data files inside the folder, see Data section):

   `python BaseSetup/base_model_lightgcn.py`

For more guidance, see `docs/GETTING_STARTED.md` and `docs/TRAINING.md`.


Repository Structure
--------------------

- `BaseSetup/`: Base LightGCN model and training script.
- `BaseSetupWithPerfOptimisations/`: Base model with mixed precision and performance tweaks.
- `FeatureAwareInitialization/`: Feature-aware embedding initialization (needs `spotify_mpd_audio_features.db`).
- `FeatureAwareMessagePassing/`: Feature-aware initialization + message passing (needs `spotify_mpd_audio_features.db`).
- `MetricsCollection/`: Saved metrics, embeddings, and experiment artifacts. (Reference only.)
- `FinalPlots/`: Final visualizations. (Reference only.)
- `hyperparameters.txt`: Legacy snapshot; not authoritative. See code and docs instead.
- `docs/`: Setup, training, and inference guides.
- `requirements.txt`: Python package dependencies for running the models.
- `requirements.py`: Guided installer to set up torch/PyG and repo deps.
- `hyperparameters.txt`: Consolidated hyperparameters pulled from scripts.


Data
----

The training scripts in each folder expect the following prepared files to exist alongside the scripts:

- `data_object.pt`: PyTorch Geometric `Data` object containing the graph.
- `dataset_stats.json`: contains `{`"num_playlists"`: ..., `"num_nodes"`: ...}`.
- `playlist_info.json` and `song_info.json`: metadata used for analysis/feature mapping.

Feature-aware variants additionally require:

- `spotify_mpd_audio_features.db`: SQLite database with audio features (columns like `danceability`, `energy`, `tempo`, etc.).

If starting from raw Spotify MPD data, see `BaseSetup/preprocess.py` for the intended preprocessing flow (K-core via SNAP, re-indexing, and conversion to PyG `Data`). Preprocessing is optional if you already have the files listed above.


Training
--------

Each variant is a runnable Python script. Common defaults are defined at the bottom of each script (epochs, batch size, layers, embedding dim, etc.).

- Base Setup:
  `python BaseSetup/base_model_lightgcn.py`

- Performance-Optimized:
  `python BaseSetupWithPerfOptimisations/base_model_lightgcn_with_perf_optimisations.py`

- Feature-Aware Initialization:
  `python FeatureAwareInitialization/base_model_lightgcn_with_feature_aware_embedding_Initialization.py`

- Feature-Aware Message Passing:
  `python FeatureAwareMessagePassing/base_model_lightgcn_with_feature_aware_embedding_Initialization_and_feature_aware_message_passing.py`

Notes:
- GPU is automatically used if available (`torch.cuda.is_available()`). The perf-optimized variant uses mixed precision when supported.
- Validation/Test splits are created via `RandomLinkSplit` inside each script.
- The base script can optionally save intermediate embeddings during validation (`embeddings/` folder).

See `docs/TRAINING.md` for detailed tips, expected console logs, and performance notes.


Inference
---------

The evaluation methods in each script compute recall@k and include all logic needed to produce ranked recommendations. A minimal recipe to get top-k song recommendations for a given playlist is provided in `docs/INFERENCE.md`. In short:

1) Load the prepared graph (`data_object.pt`, `dataset_stats.json`).
2) Recreate the model with the same hyperparameters as training.
3) Run `gnn.gnn_propagation()` to obtain multi-scale embeddings.
4) Compute scores via dot product and take `topk`, excluding songs already linked to the playlist.

If you saved model weights (some variants include checkpoints), you can load them with `gnn.load_state_dict(torch.load("path/to/checkpoint.pt"))` before inference.


Hyperparameters
---------------

- A concise, consolidated view lives in `hyperparameters.txt` (per variant). For full context and the source of truth, see the bottom sections of each training script where hyperparameters are defined and printed.


What To Ignore (Repo Browsing)
-------------------------------

- `FinalPlots/` and `MetricsCollection/` contain plots and experiment artifacts and are not required to run the code.
- `hyperparameters.txt` is a legacy dump; refer to code + docs for authoritative hyperparameters.


Reproducibility
---------------

- Scripts set seeds via `torch_geometric.seed_everything(5)` for determinism where possible.
- PyTorch and PyG introduce minor nondeterminism on GPU; see PyTorch docs for strict determinism options if needed.


Citation
--------

If you reference this work in academic writing, please cite the corresponding thesis/report and/or this repository.
