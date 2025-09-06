Getting Started
===============

This guide helps you set up a clean environment and install the correct dependencies to run the LightGCN variants in this repository.


Prerequisites
-------------

- Python 3.10 or 3.11 recommended.
- NVIDIA GPU + CUDA (optional, recommended for training speed). CPU also works.


Create an Environment
---------------------

- venv (Windows):
  - `python -m venv .venv`
  - `.\\.venv\\Scripts\\activate`

- venv (macOS/Linux):
  - `python -m venv .venv`
  - `source .venv/bin/activate`

- Conda:
  - `conda create -n lightgcn python=3.10 -y`
  - `conda activate lightgcn`


Install PyTorch
---------------

Install PyTorch first, matching your OS, Python version, and desired CUDA toolkit:

- Official selector: https://pytorch.org/get-started/locally/
- Example (CUDA 12.1):
  - `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio`
- CPU-only:
  - `pip install torch torchvision torchaudio`


Install PyTorch Geometric (PyG)
-------------------------------

PyG wheels depend on the exact `torch` and CUDA versions. Follow the official matrix:

- https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

Typical sequence (example — adjust versions to match your installed torch/CUDA):

- `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html`
- `pip install torch_geometric`

If installation fails, re-check the torch version and CUDA wheel index URL and pick the matching line from the PyG install page.


Install Project Requirements
----------------------------

After torch and PyG are installed:

- `pip install -r requirements.txt`

This installs common utilities (numpy, matplotlib, scikit-learn, ipython, tqdm). The preprocessing script optionally uses SNAP; see below.


Git LFS (Large Files)
---------------------

The repository stores large artifacts (e.g., `spotify_mpd_audio_features.db`) via Git LFS. If you cloned the repo and don’t see the real `.db` file (only a small pointer file), install and pull via LFS:

- Install LFS:
  - macOS: `brew install git-lfs`
  - Linux: `sudo apt-get install git-lfs` (or see https://git-lfs.com)
  - Windows: `winget install Git.LFS` or `choco install git-lfs`
- Initialize in this repo: `git lfs install`
- Pull binaries: `git lfs pull` (or `git lfs fetch --all && git lfs pull`)
- Verify: `git lfs ls-files` lists tracked files; `ls -lh FeatureAwareInitialization/spotify_mpd_audio_features.db` shows the real size.

Note: GitHub ZIP downloads do not include LFS binaries. Always use `git clone` and then run `git lfs pull`.


Optional: Preprocessing Dependencies
------------------------------------

`BaseSetup/preprocess.py` references the SNAP library and tqdm. If you plan to run preprocessing from raw Spotify data:

- `pip install tqdm`
- SNAP (Python module `snap`) can be installed via `pip install snap-stanford` or from source. Installation varies by platform; consult SNAP docs if needed.


Run Preprocessing (optional)
----------------------------

If you start from the raw Spotify MPD JSON files, use `BaseSetup/preprocess.py` to build the K-core graph and produce the files needed by the training scripts.

- Place the MPD `.json` files in a folder (for example, `BaseSetup/data`).
- Open `BaseSetup/preprocess.py` and set the top-level variables:
  - `data_dir`: folder with MPD files
  - `NUM_FILES_TO_USE`: number of MPD files to include
  - `save_dir`: output directory (often `'.'` to save next to the training script)
  - `K`: K-core value
- Run: `python BaseSetup/preprocess.py`

Outputs include:
- `data_object.pt`: PyG graph
- `playlist_info.json` and `song_info.json`: metadata for analysis

The training scripts also expect `dataset_stats.json` with `{ "num_playlists": ..., "num_nodes": ... }` next to the script. If you regenerate the dataset, update or recreate this file accordingly.


Next Steps
----------

- See `docs/TRAINING.md` to train/evaluate any of the four variants.
- See `docs/INFERENCE.md` for a minimal, clear recipe to produce top-k recommendations for a playlist.
