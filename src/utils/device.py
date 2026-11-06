"""
Hardware and device detection utility for P5 Vehicle Recognition.
Provides robust CPU/CUDA detection, VRAM budgeting, and hardware telemetry.
"""

from __future__ import annotations

import logging
import platform

import psutil
import torch

logger = logging.getLogger(__name__)


def get_device(requested_device: str = "auto") -> torch.device:
    """
    Resolve and return a torch.device based on availability and request.

    Args:
        requested_device: 'auto', 'cuda', 'cpu', or explicit torch device string.

    Returns:
        torch.device instance.
    """
    req = requested_device.lower().strip()
    if req in ("auto", "cuda"):
        if torch.cuda.is_available():
            dev = torch.device("cuda:0")
            logger.info("Using GPU device: %s (%s)", dev, torch.cuda.get_device_name(0))
            return dev
        if req == "cuda":
            logger.warning("CUDA requested but not available. Falling back to CPU.")
    return torch.device("cpu")


def get_hardware_telemetry() -> dict:
    """
    Gather complete hardware, OS, and compute environment specifications.

    Returns:
        Dictionary with hardware telemetry details.
    """
    vm = psutil.virtual_memory()
    info: dict = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python_version": platform.python_version(),
        "cpu_model": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "ram_available_gb": round(vm.available / (1024**3), 2),
        "ram_percent_used": vm.percent,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        gpu_idx = 0
        props = torch.cuda.get_device_properties(gpu_idx)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(gpu_idx)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_compute_capability": f"{props.major}.{props.minor}",
                "gpu_total_vram_gb": round(props.total_memory / (1024**3), 2),
                "gpu_free_vram_gb": round(free_bytes / (1024**3), 2),
                "gpu_allocated_vram_gb": round(
                    torch.cuda.memory_allocated(gpu_idx) / (1024**3), 2
                ),
                "cuda_version": torch.version.cuda,
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_version": torch.backends.cudnn.version()
                if torch.backends.cudnn.is_available()
                else None,
            }
        )
    else:
        info.update(
            {
                "gpu_name": None,
                "gpu_total_vram_gb": 0.0,
                "gpu_free_vram_gb": 0.0,
                "cuda_version": None,
            }
        )

    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    telemetry = get_hardware_telemetry()
    print("=" * 60)
    print("HARDWARE & RUNTIME TELEMETRY")
    print("=" * 60)
    for k, v in telemetry.items():
        print(f"  {k:25s}: {v}")
    dev = get_device("auto")
    print(f"  {'resolved_device':25s}: {dev}")
    print("=" * 60)
