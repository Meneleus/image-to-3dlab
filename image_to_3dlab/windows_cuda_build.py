"""Shared Windows MSVC / CUDA env for extension builds (TRELLIS + Hunyuan).

Torch's ``cpp_extension`` refuses to build when a Visual C++ environment is
active unless ``DISTUTILS_USE_SDK=1``. CUDA 12.8+/13.x CCCL also needs the
conformant MSVC preprocessor (``/Zc:preprocessor``). Env carries only Zc —
not a second ``/std:`` (that races with setup.py / torch under nvcc
``--use-local-env`` and causes cl D9025 flip-flops).
"""

from __future__ import annotations

import os
from pathlib import Path

from image_to_3dlab import host

WIN_CL_FLAGS = "/Zc:preprocessor /Zc:__cplusplus"
# nvcc must never see bare /Zc: or /std: — those look like input files.
# Host MSVC flags go through -Xcompiler=; device standard is -std=c++20.
WIN_NVCC_FLAGS = (
    "-allow-unsupported-compiler "
    "-std=c++20 "
    "-Xcompiler=/std:c++20 "
    "-Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:__cplusplus"
)


def windows_cuda_build_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for building CUDA extensions on Windows with CUDA 12.8+/13.x + MSVC.

    Sets DISTUTILS_USE_SDK and host flags via ``CL`` only. Does *not* put bare
    MSVC tokens in ``CXXFLAGS`` (torch/ninja often forward those to nvcc, which
    then treats ``/Zc:…`` / ``/std:…`` as extra input files).
    No-op on non-Windows hosts.
    """
    env = dict(base if base is not None else os.environ)
    if host.os_family() != "windows":
        return env
    env["DISTUTILS_USE_SDK"] = "1"
    env["CL"] = WIN_CL_FLAGS
    # Drop any caller/user CXXFLAGS that would leak bare /Zc or /std onto nvcc.
    env.pop("CXXFLAGS", None)
    env["NVCC_FLAGS"] = WIN_NVCC_FLAGS
    env["NVCC_PREPEND_FLAGS"] = WIN_NVCC_FLAGS
    return env


def extension_build_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Full env for a pip/uv install of a CUDA extension (PATH/CUDA_HOME + Windows flags)."""
    env = windows_cuda_build_env(base)
    nvcc = host.find_nvcc()
    if nvcc:
        env["PATH"] = os.pathsep.join([str(Path(nvcc).parent), env.get("PATH", "")])
        env.setdefault("CUDA_HOME", str(Path(nvcc).parent.parent))
        env.setdefault("CUDA_PATH", str(Path(nvcc).parent.parent))
    return env
