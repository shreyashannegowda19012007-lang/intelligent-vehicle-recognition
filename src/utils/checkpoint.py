"""
Model Checkpoint Reassembly and Management Utilities.
Handles automatic reconstruction of chunked model checkpoints for git compliance.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_convnext_checkpoint(root: Path | str | None = None) -> Path:
    """
    Ensure models/classification/convnext_best.pt exists.
    If absent, reassemble from git-tracked chunks (.part0, .part1).

    Returns:
        Path to resolved convnext_best.pt checkpoint.
    """
    if root is None:
        try:
            from src.utils.config import find_project_root
            root_path = find_project_root()
        except Exception:
            root_path = Path(__file__).resolve().parent.parent.parent
    else:
        root_path = Path(root).resolve()

    target = root_path / "models" / "classification" / "convnext_best.pt"
    if target.is_file() and target.stat().st_size > 50 * 1024 * 1024:
        return target

    p0 = root_path / "models" / "classification" / "convnext_best.pt.part0"
    p1 = root_path / "models" / "classification" / "convnext_best.pt.part1"

    if p0.is_file() and p1.is_file():
        logger.info("Reassembling ConvNeXt-Tiny weights from split chunks (%s, %s)...", p0.name, p1.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f_out:
            with open(p0, "rb") as f0:
                f_out.write(f0.read())
            with open(p1, "rb") as f1:
                f_out.write(f1.read())
        logger.info("Successfully reassembled %s (%.2f MB).", target.name, target.stat().st_size / (1024 * 1024))

    return target


@functools.lru_cache(maxsize=1)
def get_deployment_diagnostics(root_str: str | None = None) -> dict[str, Any]:
    """
    Compute developer and evaluator deployment diagnostics.
    Returns git commit, PyTorch version, active device, model presence,
    file sizes, and SHA256 fingerprints without exposing sensitive env vars.
    """
    import torch
    from src.utils.device import get_device

    if root_str is None:
        try:
            from src.utils.config import find_project_root
            root = find_project_root()
        except Exception:
            root = Path(__file__).resolve().parent.parent.parent
    else:
        root = Path(root_str).resolve()

    # Reassemble checkpoint if absent
    ensure_convnext_checkpoint(root)

    # Git commit extraction
    commit_id = "047cee7"
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=root, timeout=2)
        if res.returncode == 0 and res.stdout.strip():
            commit_id = res.stdout.strip()
    except Exception:
        pass

    def _file_info(p: Path) -> dict[str, Any]:
        if not p.is_file():
            return {
                "exists": False,
                "filename": p.name,
                "size_mb": 0.0,
                "sha256": "MISSING",
                "status": "OFFLINE (Missing on disk)",
            }
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return {
            "exists": True,
            "filename": p.name,
            "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
            "sha256": h.hexdigest()[:16],
            "status": "ONLINE (Loaded)",
        }

    det_p = root / "models" / "detection" / "best.pt"
    cls_p = root / "models" / "classification" / "convnext_best.pt"
    svm_p = root / "models" / "classical" / "svm" / "linear_svm_hog_color_lbp_pca128_model.joblib"
    pipe_p = root / "models" / "classical" / "scalers" / "linear_svm_hog_color_lbp_pca128_pipeline.joblib"

    return {
        "git_commit": commit_id,
        "pytorch_version": torch.__version__,
        "device": str(get_device("auto")),
        "class_count": 196,
        "detector": _file_info(det_p),
        "classifier": _file_info(cls_p),
        "classical_svm": _file_info(svm_p),
        "classical_pipeline": _file_info(pipe_p),
        "preprocessing": {
            "detector_input": "640x640 (Ultralytics Letterbox)",
            "classifier_input": "320x320 (Bicubic + ImageNet Norm)",
            "classical_input": "128x128 (HOG + LBP + Color Hist)",
        },
    }
