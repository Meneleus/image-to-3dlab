#!/usr/bin/env python3
"""Install official TRELLIS.2 for NVIDIA (Linux or Windows) into vendor/trellis2-cuda.

Clones https://github.com/microsoft/TRELLIS.2, creates a Python 3.11 venv with CUDA
PyTorch, installs the basic Python deps, then builds the CUDA extensions the pipeline
needs (nvdiffrast, nvdiffrec, cumesh, flexgemm, o-voxel). flash-attn is best-effort:
when it cannot build, the generate wrapper falls back to PyTorch SDPA.

Microsoft's README only claims Linux testing. On Windows the same steps run when the
CUDA toolkit and Visual Studio C++ build tools are present; if an extension fails to
compile, this says so and stops without claiming success.

Weights (~14 GB TRELLIS.2-4B + gated DINOv3) are *not* fetched here — they arrive on
the first generation run, same as the Mac bootstrap. BRIA RMBG-2.0 is never installed.

`AGENTS.md`: name the backend, route and size, and require a yes (or `--yes`).

    python scripts/bootstrap_trellis_cuda.py
    python scripts/bootstrap_trellis_cuda.py --yes
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from image_to_3dlab import host
from image_to_3dlab.cuda_routes import TRELLIS_CUDA, venv_python

UPSTREAM = "https://github.com/microsoft/TRELLIS.2.git"
VENDOR = TRELLIS_CUDA

# Rough install footprint excluding weights (clone + compiled extensions).
CODE_GB = 2.5

EXTENSIONS = (
    ("nvdiffrast", "https://github.com/NVlabs/nvdiffrast.git", "v0.4.0"),
    ("nvdiffrec", "https://github.com/JeffreyXiang/nvdiffrec.git", "renderutils"),
    ("CuMesh", "https://github.com/JeffreyXiang/CuMesh.git", None),
    ("FlexGEMM", "https://github.com/JeffreyXiang/FlexGEMM.git", None),
)

BASIC = [
    "imageio", "imageio-ffmpeg", "tqdm", "easydict", "opencv-python-headless",
    "ninja", "trimesh", "transformers", "tensorboard", "pandas", "lpips",
    "zstandard", "kornia", "timm", "Pillow", "huggingface_hub", "rembg",
    "fast_simplification",
]


def announcement() -> str:
    family = host.os_family()
    return "\n".join([
        "",
        "About to install:",
        "",
        "  backend: TRELLIS.2 (official CUDA)",
        f"  route:   clone + CUDA extensions into {VENDOR.relative_to(REPO)}/",
        f"  code:    ~{CODE_GB:.1f} GB (no model weights yet)",
        "  weights: ~14 GB TRELLIS.2-4B + gated DINOv3 on first Generate run",
        f"  host:    {family} NVIDIA",
        "",
        "  licence: MIT (code + weights); DINOv3 encoder is separately gated",
        "           https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m",
        "",
        "  note:    Microsoft tests TRELLIS.2 on Linux. Windows needs the CUDA",
        "           toolkit and Visual Studio C++ build tools for the extensions.",
        "",
    ])


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_clone() -> None:
    if (VENDOR / ".git").is_dir():
        print(f"Already cloned: {VENDOR}", flush=True)
        return
    VENDOR.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--recursive", UPSTREAM, str(VENDOR)])


def ensure_venv() -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required. Install from https://docs.astral.sh/uv/")
    py = venv_python(VENDOR)
    if not py.is_file():
        run([uv, "venv", "--python", "3.11", str(VENDOR / ".venv")])
    return py


def install_torch(py: Path) -> None:
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


def install_basic(py: Path) -> None:
    uv = shutil.which("uv")
    run([uv, "pip", "install", "--python", str(py), *BASIC])
    run([uv, "pip", "install", "--python", str(py),
         "git+https://github.com/EasternJournalist/utils3d.git"
         "@9a4eb15e4021b67b12c460c7057d642626897ec8"])


def install_extension(py: Path, name: str, url: str, ref: str | None,
                      work: Path) -> None:
    target = work / name
    if not (target / ".git").is_dir():
        cmd = ["git", "clone", "--recursive", url, str(target)]
        run(cmd)
        if ref:
            run(["git", "checkout", ref], cwd=target)
    uv = shutil.which("uv")
    env = dict(os.environ)
    nvcc = host.find_nvcc()
    if nvcc:
        env["PATH"] = os.pathsep.join([str(Path(nvcc).parent), env.get("PATH", "")])
        env.setdefault("CUDA_HOME", str(Path(nvcc).parent.parent))
    run([uv, "pip", "install", "--python", str(py), "--no-build-isolation", str(target)],
        env=env)


def install_extensions(py: Path) -> None:
    work = VENDOR / ".i2l-build"
    work.mkdir(parents=True, exist_ok=True)
    for name, url, ref in EXTENSIONS:
        print(f"\nBuilding {name}...", flush=True)
        try:
            install_extension(py, name, url, ref, work)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Failed to build {name}. TRELLIS.2 needs a working CUDA toolkit "
                f"({'and Visual Studio C++ build tools ' if host.os_family() == 'windows' else ''}"
                f"matching this machine). See docs/WINDOWS.md.\n{exc}"
            ) from exc
    # o-voxel ships inside the TRELLIS.2 clone
    o_voxel = VENDOR / "o-voxel"
    if o_voxel.is_dir():
        print("\nBuilding o-voxel...", flush=True)
        uv = shutil.which("uv")
        env = dict(os.environ)
        nvcc = host.find_nvcc()
        if nvcc:
            env["PATH"] = os.pathsep.join([str(Path(nvcc).parent), env.get("PATH", "")])
            env.setdefault("CUDA_HOME", str(Path(nvcc).parent.parent))
        try:
            run([uv, "pip", "install", "--python", str(py), "--no-build-isolation",
                 str(o_voxel)], env=env)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"Failed to build o-voxel: {exc}") from exc


def try_flash_attn(py: Path) -> None:
    uv = shutil.which("uv")
    print("\nTrying flash-attn (optional; SDPA works if this fails)...", flush=True)
    done = subprocess.run(
        [uv, "pip", "install", "--python", str(py), "flash-attn==2.7.3",
         "--no-build-isolation"],
        check=False,
    )
    if done.returncode != 0:
        print("flash-attn not installed; generate will use ATTN_BACKEND=sdpa.", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true",
                        help="Skip the confirmation. For the viewer and for agents.")
    args = parser.parse_args(argv)

    if host.host_platform() != host.NVIDIA:
        print("TRELLIS CUDA installs on Linux/Windows with an NVIDIA GPU. "
              "On Apple Silicon use scripts/bootstrap_trellis_space_macos.py. "
              "Nothing downloaded.")
        return 1

    print(announcement())
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            print("Refusing to install without --yes when there is nobody to ask.")
            return 1
        if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
            print("Nothing downloaded.")
            return 1

    if not host.find_nvcc():
        print("Warning: nvcc not found. Extension builds will likely fail. "
              "Install the CUDA toolkit (and on Windows, Visual Studio C++ build tools).",
              flush=True)

    ensure_clone()
    py = ensure_venv()
    install_torch(py)
    install_basic(py)
    install_extensions(py)
    try_flash_attn(py)
    print("\nDone. Pick TRELLIS.2 in Generate 3D. First run downloads ~14 GB of weights "
          "(DINOv3 is gated — run `hf auth login` first).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
