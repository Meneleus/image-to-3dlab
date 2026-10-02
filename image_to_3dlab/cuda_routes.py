"""Host-aware vendor paths for TRELLIS.2 and Hunyuan3D CUDA vs Mac ports.

One catalogue id per product (`trellis`, `hunyuan_xiong`). On Apple Silicon the lab uses
the Metal/MLX ports; on NVIDIA it uses the official CUDA checkouts under `vendor/`.
"""

from __future__ import annotations

import os
from pathlib import Path

from image_to_3dlab.host import APPLE, NVIDIA, host_platform

REPO = Path(__file__).resolve().parents[1]

TRELLIS_MAC = REPO / "vendor" / "trellis-space-mac"
TRELLIS_CUDA = REPO / "vendor" / "trellis2-cuda"
TRELLIS_MAC_WRAPPER = REPO / "scripts" / "trellis_space_generate.py"
TRELLIS_CUDA_WRAPPER = REPO / "scripts" / "trellis_cuda_generate.py"
TRELLIS_MAC_BOOTSTRAP = REPO / "scripts" / "bootstrap_trellis_space_macos.py"
TRELLIS_CUDA_BOOTSTRAP = REPO / "scripts" / "bootstrap_trellis_cuda.py"

HUNYUAN_MLX_SHAPE = REPO / "hunyuan_mlx" / "shape"
HUNYUAN_MLX_PAINT = REPO / "hunyuan_mlx" / "paint"
HUNYUAN_CUDA = REPO / "vendor" / "hunyuan3d-cuda"
HUNYUAN_MLX_WRAPPER = REPO / "scripts" / "hunyuan_mlx_xiong_generate.py"
HUNYUAN_CUDA_WRAPPER = REPO / "scripts" / "hunyuan_cuda_generate.py"
HUNYUAN_CUDA_BOOTSTRAP = REPO / "scripts" / "bootstrap_hunyuan_cuda.py"


def venv_python(project: Path) -> Path:
    if os.name == "nt":
        return project / ".venv" / "Scripts" / "python.exe"
    return project / ".venv" / "bin" / "python"


def trellis_route(platform_id: str | None = None) -> str:
    """`mac` or `cuda` for the TRELLIS.2 stack this host should use."""
    host = platform_id or host_platform()
    if host == APPLE:
        return "mac"
    if host == NVIDIA:
        return "cuda"
    return "none"


def hunyuan_route(platform_id: str | None = None) -> str:
    """`mlx` or `cuda` for the Hunyuan3D stack this host should use."""
    host = platform_id or host_platform()
    if host == APPLE:
        return "mlx"
    if host == NVIDIA:
        return "cuda"
    return "none"


def trellis_vendor(platform_id: str | None = None) -> Path:
    return TRELLIS_MAC if trellis_route(platform_id) == "mac" else TRELLIS_CUDA


def trellis_python(platform_id: str | None = None) -> Path:
    return venv_python(trellis_vendor(platform_id))


def trellis_wrapper(platform_id: str | None = None) -> Path:
    return TRELLIS_MAC_WRAPPER if trellis_route(platform_id) == "mac" else TRELLIS_CUDA_WRAPPER


def trellis_bootstrap(platform_id: str | None = None) -> Path:
    return TRELLIS_MAC_BOOTSTRAP if trellis_route(platform_id) == "mac" else TRELLIS_CUDA_BOOTSTRAP


def trellis_build_present(platform_id: str | None = None) -> bool:
    return trellis_python(platform_id).is_file() and trellis_wrapper(platform_id).is_file()


def hunyuan_python(platform_id: str | None = None) -> Path:
    if hunyuan_route(platform_id) == "mlx":
        return venv_python(HUNYUAN_MLX_SHAPE)
    return venv_python(HUNYUAN_CUDA)


def hunyuan_wrapper(platform_id: str | None = None) -> Path:
    if hunyuan_route(platform_id) == "mlx":
        return HUNYUAN_MLX_WRAPPER
    return HUNYUAN_CUDA_WRAPPER


def hunyuan_build_present(platform_id: str | None = None) -> bool:
    if hunyuan_route(platform_id) == "mlx":
        return (venv_python(HUNYUAN_MLX_SHAPE).is_file()
                and venv_python(HUNYUAN_MLX_PAINT).is_file())
    return hunyuan_python(platform_id).is_file() and (HUNYUAN_CUDA / "hy3dshape").is_dir()
