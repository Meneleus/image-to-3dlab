#!/usr/bin/env python3
"""Install official Hunyuan3D-2.1 for NVIDIA into vendor/hunyuan3d-cuda.

Clones https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 (supports Linux and Windows),
creates a Python 3.11 venv with CUDA PyTorch, installs a curated requirements set
(skipping deepspeed/bpy which break plain Windows installs), builds the custom
rasterizer, and optionally downloads the ~10 GB shape+paint weights.

On Windows, CUDA extension installs reuse the same MSVC gate as TRELLIS:
``DISTUTILS_USE_SDK=1``, ``/Zc:preprocessor``, and the C++20 setup.py hook —
needed when the x64 Native Tools shell is already active (torch otherwise
refuses the build).

`AGENTS.md`: name the backend, route and size, and require a yes (or `--yes`).

    python scripts/bootstrap_hunyuan_cuda.py
    python scripts/bootstrap_hunyuan_cuda.py --yes
    python scripts/bootstrap_hunyuan_cuda.py --yes --code-only
    python scripts/bootstrap_hunyuan_cuda.py --yes --weights-only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# TRELLIS bootstrap owns the setup.py C++20 / CCCL rewrite helpers we reuse.
sys.path.insert(0, str(REPO / "scripts"))

import bootstrap_trellis_cuda as trellis_boot

from image_to_3dlab import host
from image_to_3dlab.cuda_routes import HUNYUAN_CUDA, venv_python
from image_to_3dlab.windows_cuda_build import extension_build_env

UPSTREAM = "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
VENDOR = HUNYUAN_CUDA
WEIGHTS_REPO = "tencent/Hunyuan3D-2.1"
WEIGHTS_GB = 10.0
CODE_GB = 1.5
REALESRGAN_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
)

# Curated: upstream requirements.txt pins deepspeed/bpy/tb_nightly which fight Windows
# and the lab's own Blender install. Torch is installed separately from the CUDA index.
# pymeshlab: hy3dshape.postprocessors imports it; upstream pins 2022.2.post3 (no
# Windows py3.11 wheel). 2025.7.post1 is the wheel that works on REDFURY.
# rembg does not always pull onnxruntime; install it explicitly into this vendor venv.
PIP_PACKAGES = [
    "ninja", "pybind11",
    "transformers==4.46.0", "diffusers==0.30.0", "accelerate==1.1.1",
    "huggingface-hub==0.30.2", "safetensors==0.4.4",
    "numpy<2", "scipy", "einops", "pandas",
    "opencv-python", "imageio", "scikit-image", "rembg",
    "trimesh", "pymeshlab==2025.7.post1", "pygltflib", "xatlas", "omegaconf", "pyyaml",
    "tqdm", "psutil", "timm", "torchmetrics", "pydantic",
]
# pymeshlab's Windows wheels need the MSVC runtime DLLs beside the interpreter.
WINDOWS_PIP_PACKAGES = ("msvc-runtime",)


def runtime_packages() -> list[str]:
    """Packages for vendor/hunyuan3d-cuda/.venv (not the lab root .venv)."""
    packages = list(PIP_PACKAGES)
    if host.os_family() == "windows":
        # GPU ORT for rembg CUDA EP; same import name as onnxruntime.
        packages.append("onnxruntime-gpu")
        packages.extend(WINDOWS_PIP_PACKAGES)
    else:
        packages.append("onnxruntime")
    return packages


def announcement(code: bool = True, weights: bool = True) -> str:
    lines = [
        "",
        "About to install:",
        "",
        "  backend: Hunyuan3D-2.1 (official CUDA)",
        f"  route:   clone + CUDA rasterizer into {VENDOR.relative_to(REPO)}/",
    ]
    if code:
        lines.append(f"  code:    ~{CODE_GB:.1f} GB checkout + compiled extensions")
    if weights:
        lines.append(f"  weights: ~{WEIGHTS_GB:.1f} GB from {WEIGHTS_REPO}")
        lines.append("           (shape dit-v2-1 + paint PBR + RealESRGAN)")
    lines += [
        "",
        "  licence: Tencent Hunyuan Community License — NOT licensed for use in",
        "           the EU, UK or South Korea. Read it before downloading.",
        f"           https://huggingface.co/{WEIGHTS_REPO}",
        "",
    ]
    return "\n".join(lines)


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_clone() -> None:
    if (VENDOR / ".git").is_dir():
        print(f"Already cloned: {VENDOR}", flush=True)
        return
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", UPSTREAM, str(VENDOR)])


def ensure_venv() -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required. Install from https://docs.astral.sh/uv/")
    py = venv_python(VENDOR)
    if not py.is_file():
        run([uv, "venv", "--python", "3.11", str(VENDOR / ".venv")])
    return py


def patch_custom_rasterizer_msvc_narrowing(raster: Path) -> None:
    """MSVC C2398 narrowing + long→int64_t LibTorch ABI fixes for custom_rasterizer."""
    if host.os_family() != "windows":
        return
    script = REPO / "scripts" / "patch_hunyuan_rasterizer_msvc_narrowing.py"
    print(f"Patching custom_rasterizer for MSVC ({script.name})...", flush=True)
    run([sys.executable, str(script), "--root", str(raster)])


def install_cuda_extension(py: Path, target: Path, label: str) -> None:
    """Build + install a CUDA extension package (non-editable; Windows MSVC-safe)."""
    uv = shutil.which("uv")
    setup_py = target / "setup.py"
    if setup_py.is_file():
        trellis_boot.ensure_windows_extension_setup(setup_py)
        trellis_boot.clean_extension_build_artifacts(target)
    env = extension_build_env()
    print(f"Building {label}...", flush=True)
    # Non-editable: same shape as TRELLIS extensions. Editable is fragile on MSVC
    # and unnecessary — generate uses the package installed into this Hunyuan venv.
    run(
        [uv, "pip", "install", "--python", str(py), "--no-build-isolation",
         "--force-reinstall", "--no-deps", str(target)],
        env=env,
    )


def install_code(py: Path) -> None:
    uv = shutil.which("uv")
    index = host.torch_cuda_index(host.driver_cuda_version())
    if not index:
        raise SystemExit(
            "No PyTorch CUDA build matches this NVIDIA driver (need CUDA 12.8+). "
            "Update the driver, then run this again."
        )
    print(f"Installing PyTorch with CUDA from {index}", flush=True)
    run([uv, "pip", "install", "--python", str(py), "--index-url", index,
         "torch", "torchvision"])
    # `--python` is the Hunyuan vendor venv (vendor/hunyuan3d-cuda/.venv), not
    # the lab root .venv — Generate 3D uses hunyuan_python() from that checkout.
    run([uv, "pip", "install", "--python", str(py), *runtime_packages()])

    if host.os_family() == "windows":
        print("\nPatching torch.utils.cpp_extension for C++20...", flush=True)
        trellis_boot.patch_torch_cpp_extension_cxx20(py)

    raster = VENDOR / "hy3dpaint" / "custom_rasterizer"
    if raster.is_dir():
        patch_custom_rasterizer_msvc_narrowing(raster)
        try:
            install_cuda_extension(py, raster, "custom_rasterizer")
        except subprocess.CalledProcessError as exc:
            tip = ""
            if host.os_family() == "windows":
                tip = (" Install Visual Studio Build Tools with C++, and a CUDA toolkit "
                       "matching PyTorch. From a shell where `cl` works — e.g. "
                       "`VsDevCmd.bat -arch=amd64` — then re-run. See docs/WINDOWS.md.")
            raise SystemExit(f"custom_rasterizer failed to build.{tip}") from exc

    # DifferentiableRenderer: prefer setup.py / pip when present; else compile script.
    diff = VENDOR / "hy3dpaint" / "DifferentiableRenderer"
    setup = diff / "setup.py"
    if setup.is_file():
        try:
            install_cuda_extension(py, diff, "DifferentiableRenderer")
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"DifferentiableRenderer failed to build: {exc}") from exc
    elif (diff / "compile_mesh_painter.sh").is_file() and host.os_family() != "windows":
        print("Compiling DifferentiableRenderer via shell script...", flush=True)
        run(["bash", "compile_mesh_painter.sh"], cwd=diff)

    ckpt = VENDOR / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"
    if not ckpt.is_file():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        print(f"Fetching RealESRGAN weights -> {ckpt}", flush=True)
        urllib.request.urlretrieve(REALESRGAN_URL, ckpt)


def install_weights() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub missing; re-run without --weights-only first") from exc
    print(f"\nFetching {WEIGHTS_REPO} (~{WEIGHTS_GB:.1f} GB)...", flush=True)
    snapshot_download(
        WEIGHTS_REPO,
        allow_patterns=[
            "hunyuan3d-dit-v2-1/*",
            "hunyuan3d-paintpbr-v2-1/*",
            "README.md",
            "LICENSE*",
        ],
        local_dir=str(VENDOR / "weights"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    args = parser.parse_args(argv)

    if host.host_platform() != host.NVIDIA:
        print("Hunyuan CUDA installs on Linux/Windows with an NVIDIA GPU. "
              "On Apple Silicon use the Hunyuan3D-MLX route. Nothing downloaded.")
        return 1

    code = not args.weights_only
    weights = not args.code_only
    print(announcement(code=code, weights=weights))
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to install without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if code:
        ensure_clone()
        py = ensure_venv()
        install_code(py)
    if weights:
        if not (VENDOR / ".git").is_dir():
            ensure_clone()
        install_weights()
    print("\nDone. Pick Hunyuan3D in Generate 3D.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
