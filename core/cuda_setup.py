from __future__ import annotations

import os
import sys
from pathlib import Path


def _get_nvidia_base_path() -> Path | None:
    venv_base = Path(sys.executable).parent.parent
    
    if sys.platform == "win32":
        nvidia_path = venv_base / "Lib" / "site-packages" / "nvidia"
    else:
        python_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        nvidia_path = venv_base / "lib" / python_ver / "site-packages" / "nvidia"
    
    if nvidia_path.exists():
        return nvidia_path
    
    for path in sys.path:
        candidate = Path(path) / "nvidia"
        if candidate.exists():
            return candidate
    
    return None


def _get_library_paths(nvidia_base: Path) -> list[Path]:
    if sys.platform == "win32":
        return [
            nvidia_base / "cuda_runtime" / "bin",
            nvidia_base / "cuda_runtime" / "lib" / "x64",
            nvidia_base / "cublas" / "bin",
            nvidia_base / "cudnn" / "bin",
            nvidia_base / "cuda_nvrtc" / "bin",
            nvidia_base / "cuda_nvcc" / "bin",
            nvidia_base / "cufft" / "bin",
            nvidia_base / "curand" / "bin",
        ]
    else:
        return [
            nvidia_base / "cuda_runtime" / "lib",
            nvidia_base / "cublas" / "lib",
            nvidia_base / "cudnn" / "lib",
            nvidia_base / "cuda_nvrtc" / "lib",
            nvidia_base / "cufft" / "lib",
            nvidia_base / "curand" / "lib",
        ]


def set_cuda_paths() -> bool:
    nvidia_base = _get_nvidia_base_path()
    
    if nvidia_base is None:
        return False
    
    library_paths = _get_library_paths(nvidia_base)
    existing_paths = [p for p in library_paths if p.exists()]
    
    if not existing_paths:
        return False
    
    path_strings = [str(p) for p in existing_paths]
    
    if sys.platform == "win32":
        env_var = "PATH"
        for path in existing_paths:
            try:
                os.add_dll_directory(str(path))
            except (OSError, AttributeError):
                pass
    else:
        env_var = "LD_LIBRARY_PATH"
    
    current_value = os.environ.get(env_var, "")
    new_value = os.pathsep.join(path_strings + ([current_value] if current_value else []))
    os.environ[env_var] = new_value
    
    cuda_runtime_path = nvidia_base / "cuda_runtime"
    if cuda_runtime_path.exists():
        current_cuda = os.environ.get("CUDA_PATH", "")
        new_cuda = os.pathsep.join([str(cuda_runtime_path)] + ([current_cuda] if current_cuda else []))
        os.environ["CUDA_PATH"] = new_cuda
    
    return True


def setup_cuda_if_available() -> bool:
    try:
        return set_cuda_paths()
    except Exception:
        return False