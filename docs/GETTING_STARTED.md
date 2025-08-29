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


Optional: Preprocessing Dependencies
------------------------------------

`BaseSetup/preprocess.py` references the SNAP library and tqdm. If you plan to run preprocessing from raw Spotify data:

- `pip install tqdm`
- SNAP (Python module `snap`) can be installed via `pip install snap-stanford` or from source. Installation varies by platform; consult SNAP docs if needed.


Next Steps
----------

- See `docs/TRAINING.md` to train/evaluate any of the four variants.
- See `docs/INFERENCE.md` for a minimal, clear recipe to produce top-k recommendations for a playlist.

