"""
requirements.py — guided installer for PyTorch, PyG, and repo deps

Usage examples:
  - Dry run:          python requirements.py --dry-run
  - CPU only:         python requirements.py --cpu --yes
  - CUDA 12.1:        python requirements.py --cuda cu121 --yes
  - CUDA 11.8:        python requirements.py --cuda cu118 --yes
  - Skip PyG:         python requirements.py --skip-pyg --yes

Notes:
  - This script assembles pip commands that mirror the official instructions
    for installing torch and PyTorch Geometric (PyG). It detects your installed
    torch version to pick the matching PyG wheel index, then installs the rest
    of the repo requirements from requirements.txt.
  - If something fails due to platform/CUDA/toolchain specifics, re-run with
    --dry-run to see commands and adjust manually using the official guides:
      • PyTorch: https://pytorch.org/get-started/locally/
      • PyG:     https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def sh(cmd: list[str], dry: bool = False) -> int:
    print("$", " ".join(cmd))
    if dry:
        return 0
    return subprocess.call(cmd)


def detect_torch_version() -> tuple[str | None, str]:
    try:
        import torch  # type: ignore
        ver = torch.__version__.split("+", 1)[0]
        cuda_ver = getattr(torch.version, "cuda", None)
        return ver, (cuda_ver or "cpu")
    except Exception:
        return None, "unknown"


def build_pyg_index(torch_ver: str, cuda_tag: str) -> str:
    # cuda_tag is one of: "cu121", "cu118", or "cpu"
    return f"https://data.pyg.org/whl/torch-{torch_ver}+{cuda_tag}.html"


def install_torch(cuda: str | None, yes: bool, dry: bool) -> int:
    base = [sys.executable, "-m", "pip", "install"]
    if cuda in {"cu121", "cu118"}:
        idx = f"https://download.pytorch.org/whl/{cuda}"
        cmd = base + ["--index-url", idx, "torch", "torchvision", "torchaudio"]
    else:
        # CPU or unspecified -> default PyPI
        cmd = base + ["torch", "torchvision", "torchaudio"]
    if not yes:
        print("[info] Torch install command (use --yes to execute):")
        dry = True
    return sh(cmd, dry)


def install_pyg(skip: bool, yes: bool, dry: bool) -> int:
    if skip:
        print("[info] Skipping PyG install (per --skip-pyg)")
        return 0

    torch_ver, torch_cuda = detect_torch_version()
    if not torch_ver:
        print("[warn] Torch not importable; install torch first (see --cpu/--cuda).")
        return 1

    # Map torch.version.cuda -> tag expected by PyG index
    cuda_tag = "cpu"
    if torch_cuda and torch_cuda != "cpu":
        m = re.match(r"^(\d+)\.(\d+)", torch_cuda)
        if m:
            major, minor = m.groups()
            cuda_tag = f"cu{major}{minor}"
    idx = build_pyg_index(torch_ver, cuda_tag)

    base = [sys.executable, "-m", "pip", "install"]
    cmd_core = [
        "pyg_lib",
        "torch_scatter",
        "torch_sparse",
        "torch_cluster",
        "torch_spline_conv",
        "-f",
        idx,
    ]
    if not yes:
        print("[info] PyG install command (use --yes to execute):")
        dry = True
    rc = sh(base + cmd_core, dry)
    if rc != 0:
        print("[warn] PyG core wheels may not match your torch/CUDA. Check:")
        print("       ", idx)
        return rc
    # torch_geometric itself is pure Python on PyPI
    return sh(base + ["torch_geometric"], dry)


def install_repo_requirements(yes: bool, dry: bool) -> int:
    req = Path("requirements.txt")
    if not req.exists():
        print("[warn] requirements.txt missing; skipping.")
        return 0
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    if not yes:
        print("[info] Repo requirements install command (use --yes to execute):")
        dry = True
    return sh(cmd, dry)


def main() -> int:
    p = argparse.ArgumentParser(description="Guided installer for torch, PyG, and project deps")
    cuda_grp = p.add_mutually_exclusive_group()
    cuda_grp.add_argument("--cpu", action="store_true", help="Install CPU-only torch")
    cuda_grp.add_argument("--cuda", choices=["cu121", "cu118"], help="Install torch for given CUDA toolkit")
    p.add_argument("--skip-pyg", action="store_true", help="Skip PyTorch Geometric installation")
    p.add_argument("--skip-torch", action="store_true", help="Skip torch installation")
    p.add_argument("--yes", action="store_true", help="Execute installs (omit to just preview)")
    p.add_argument("--dry-run", action="store_true", help="Alias for preview; do not execute")

    args = p.parse_args()
    dry = bool(args.dry_run)
    yes = bool(args.yes) and not dry

    print("[info] Python:", sys.version.replace("\n", " "))
    tv, tc = detect_torch_version()
    print(f"[info] Detected torch: {tv or 'not installed'} (CUDA: {tc})")

    if not args.skip_torch:
        rc = install_torch("cpu" if args.cpu else args.cuda, yes=yes, dry=dry)
        if rc != 0:
            return rc

    rc = install_pyg(skip=args.skip_pyg, yes=yes, dry=dry)
    if rc != 0:
        return rc

    rc = install_repo_requirements(yes=yes, dry=dry)
    if rc != 0:
        return rc

    print("\n[success] Installation steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

