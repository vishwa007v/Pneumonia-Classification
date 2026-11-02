"""
Device detection and hardware awareness module.

Ensures seamless operation on local machines without dedicated NVIDIA CUDA GPUs
(such as AMD Ryzen APUs with integrated Radeon graphics) as well as cloud/Kaggle
environments equipped with NVIDIA CUDA accelerators.
"""

import os
import platform
import torch


def get_compute_device() -> torch.device:
    """
    Selects the primary compute device without assuming CUDA is present.
    
    Returns:
        torch.device: CUDA device if an NVIDIA GPU is detected and supported,
                      otherwise falls back cleanly to CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_hardware_info() -> dict:
    """
    Gathers system and compute hardware diagnostics.
    
    Returns:
        dict: Diagnostics including OS, CPU count, PyTorch version, and GPU availability.
    """
    cuda_available = torch.cuda.is_available()
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu_count": os.cpu_count(),
        "cuda_available": cuda_available,
        "selected_device": "cuda" if cuda_available else "cpu",
    }

    if cuda_available:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            info["gpu_memory_total_gb"] = round(total_mem_gb, 2)
        except Exception:
            info["gpu_memory_total_gb"] = None
    else:
        info["note"] = (
            "CUDA not available or running on non-NVIDIA/integrated graphics. "
            "Safe CPU fallback active for local execution."
        )

    return info


def print_device_info():
    """
    Prints a clear, formatted summary of system compute diagnostics.
    """
    info = get_hardware_info()
    print("=" * 65)
    print(" COMPUTE ENVIRONMENT DIAGNOSTICS")
    print("=" * 65)
    print(f" Platform        : {info['platform']}")
    print(f" Processor       : {info['processor']}")
    print(f" CPU Cores       : {info['cpu_count']}")
    print(f" Python Version  : {info['python_version']}")
    print(f" PyTorch Version : {info['pytorch_version']}")
    if info["cuda_available"]:
        print(f" CUDA Status     : AVAILABLE ({info['cuda_device_count']} device(s))")
        print(f" Primary GPU     : {info['gpu_name']}")
        if info.get("gpu_memory_total_gb"):
            print(f" GPU VRAM Total  : {info['gpu_memory_total_gb']} GB")
    else:
        print(" CUDA Status     : NOT DETECTED (Running in CPU mode)")
        print(f" Notice          : {info['note']}")
    print(f" Active Device   : {info['selected_device'].upper()}")
    print("=" * 65)
