"""Mac vs CUDA vendor path selection for TRELLIS and Hunyuan."""

from __future__ import annotations

from image_to_3dlab import cuda_routes
from image_to_3dlab.host import APPLE, NVIDIA


def test_apple_gets_mac_trellis_and_mlx_hunyuan():
    assert cuda_routes.trellis_route(APPLE) == "mac"
    assert cuda_routes.hunyuan_route(APPLE) == "mlx"
    assert cuda_routes.trellis_vendor(APPLE) == cuda_routes.TRELLIS_MAC
    assert cuda_routes.trellis_wrapper(APPLE).name == "trellis_space_generate.py"
    assert cuda_routes.hunyuan_wrapper(APPLE).name == "hunyuan_mlx_xiong_generate.py"
    assert cuda_routes.trellis_bootstrap(APPLE).name == "bootstrap_trellis_space_macos.py"


def test_nvidia_gets_cuda_stacks():
    assert cuda_routes.trellis_route(NVIDIA) == "cuda"
    assert cuda_routes.hunyuan_route(NVIDIA) == "cuda"
    assert cuda_routes.trellis_vendor(NVIDIA) == cuda_routes.TRELLIS_CUDA
    assert cuda_routes.trellis_wrapper(NVIDIA).name == "trellis_cuda_generate.py"
    assert cuda_routes.hunyuan_wrapper(NVIDIA).name == "hunyuan_cuda_generate.py"
    assert cuda_routes.trellis_bootstrap(NVIDIA).name == "bootstrap_trellis_cuda.py"


def test_other_hosts_get_no_route():
    assert cuda_routes.trellis_route("other") == "none"
    assert cuda_routes.hunyuan_route("other") == "none"


def test_venv_python_spelling(monkeypatch):
    project = cuda_routes.TRELLIS_CUDA
    monkeypatch.setattr(cuda_routes.os, "name", "posix")
    assert cuda_routes.venv_python(project).as_posix().endswith(".venv/bin/python")
    monkeypatch.setattr(cuda_routes.os, "name", "nt")
    assert cuda_routes.venv_python(project).as_posix().endswith(".venv/Scripts/python.exe")
