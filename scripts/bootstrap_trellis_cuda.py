#!/usr/bin/env python3
"""Install official TRELLIS.2 for NVIDIA (Linux or Windows) into vendor/trellis2-cuda.

Clones https://github.com/microsoft/TRELLIS.2, creates a Python 3.11 venv with CUDA
PyTorch, installs the basic Python deps, then builds the CUDA extensions the pipeline
needs (nvdiffrast, nvdiffrec, cumesh, flexgemm, o-voxel). flash-attn is best-effort:
when it cannot build, the generate wrapper falls back to PyTorch SDPA.

Microsoft's README only claims Linux testing. On Windows the same steps run when the
CUDA toolkit and Visual Studio C++ build tools are present; if an extension fails to
compile, this says so and stops without claiming success.

On Windows this also applies the MSVC/CUDA flags modern toolchains need (C++20,
`/Zc:preprocessor`) — as process env and by *replacing* every hardcoded `c++17` in
extension `setup.py` files (FlexGEMM's later `/std:c++17` overrides env C++20).

Weights (~14 GB TRELLIS.2-4B + gated DINOv3) are *not* fetched here — they arrive on
the first generation run, same as the Mac bootstrap. BRIA RMBG-2.0 is never installed.

`AGENTS.md`: name the backend, route and size, and require a yes (or `--yes`).

    python scripts/bootstrap_trellis_cuda.py
    python scripts/bootstrap_trellis_cuda.py --yes
"""

from __future__ import annotations

import argparse
import os
import re
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

# CUDA 12.8+/13.x CCCL + current PyTorch headers need C++20 and MSVC's conformant
# preprocessor. FlexGEMM/CuMesh hardcode C++17 without /Zc:preprocessor. Env
# CXXFLAGS alone loses: cl warns D9025 and the later /std:c++17 from setup.py wins.
# We rewrite every c++17 in setup.py before pip install — see ensure_windows_extension_setup.
WIN_CXXFLAGS = "/std:c++20 /Zc:preprocessor /Zc:__cplusplus"
WIN_NVCC_FLAGS = (
    "-std=c++20 -allow-unsupported-compiler "
    "-Xcompiler=/std:c++20 -Xcompiler=/Zc:preprocessor -Xcompiler=/Zc:__cplusplus"
)
WIN_SETUP_MARKER = "# image-to-3dlab: windows cxx20 + Zc:preprocessor\n"
_CXX17_IN_SETUP = re.compile(r"c\+\+17", re.IGNORECASE)


def announcement() -> str:
    family = host.os_family()
    lines = [
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
    ]
    if family == "windows":
        lines += [
            "           This script sets DISTUTILS_USE_SDK, C++20, and",
            "           /Zc:preprocessor for CUDA 12.8+/13.x + modern MSVC.",
        ]
    lines.append("")
    return "\n".join(lines)


def windows_cuda_build_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for building CUDA extensions on Windows with CUDA 12.8+/13.x + MSVC.

    Sets DISTUTILS_USE_SDK (so distutils finds the VS toolchain), C++20, and the
    conformant preprocessor flag CCCL requires. No-op on non-Windows hosts.
    """
    env = dict(base if base is not None else os.environ)
    if host.os_family() != "windows":
        return env
    env["DISTUTILS_USE_SDK"] = "1"
    env["CXXFLAGS"] = WIN_CXXFLAGS
    env["CL"] = WIN_CXXFLAGS
    env["NVCC_FLAGS"] = WIN_NVCC_FLAGS
    env["NVCC_PREPEND_FLAGS"] = WIN_NVCC_FLAGS
    return env


def setup_still_has_cxx17(text: str) -> bool:
    """True if setup.py still names C++17 anywhere (compile flags must not)."""
    return _CXX17_IN_SETUP.search(text) is not None


def rewrite_setup_cxx_flags(text: str) -> str:
    """Replace every C++17 compile-flag spelling with C++20; add /Zc:preprocessor.

    Pure string rewrite used on Windows before building FlexGEMM / CuMesh / o-voxel.
    Must *replace* hardcoded c++17 — prepending c++20 via env loses (cl D9025).
    """
    # Replace concrete flag spellings first, then any leftover c++17 token.
    text = text.replace("-Xcompiler=/std:c++17", "-Xcompiler=/std:c++20")
    text = text.replace("/std:c++17", "/std:c++20")
    text = text.replace("-std=c++17", "-std=c++20")
    text = _CXX17_IN_SETUP.sub("c++20", text)

    text = _ensure_zc_preprocessor(text)
    text = _rewrite_unix_only_args_for_msvc(text)
    return text


def _ensure_zc_preprocessor(text: str) -> str:
    """Add /Zc:preprocessor on MSVC cxx and nvcc -Xcompiler lines when missing."""
    if "/Zc:preprocessor" in text:
        return text
    if "/Zc:__cplusplus" in text:
        text = text.replace(
            '"/Zc:__cplusplus"',
            '"/Zc:__cplusplus", "/Zc:preprocessor"',
        )
        text = text.replace(
            "'/Zc:__cplusplus'",
            "'/Zc:__cplusplus', '/Zc:preprocessor'",
        )
        text = text.replace(
            '"-Xcompiler=/Zc:__cplusplus"',
            '"-Xcompiler=/Zc:__cplusplus", "-Xcompiler=/Zc:preprocessor"',
        )
        text = text.replace(
            "'-Xcompiler=/Zc:__cplusplus'",
            "'-Xcompiler=/Zc:__cplusplus', '-Xcompiler=/Zc:preprocessor'",
        )
        return text
    # Windows cxx list present but no Zc flags yet (inject after /std:c++20).
    if '"/std:c++20"' in text:
        text = text.replace(
            '"/std:c++20"',
            '"/std:c++20", "/Zc:preprocessor", "/Zc:__cplusplus"',
            1,
        )
    if '"-Xcompiler=/std:c++20"' in text:
        text = text.replace(
            '"-Xcompiler=/std:c++20"',
            '"-Xcompiler=/std:c++20", "-Xcompiler=/Zc:preprocessor", '
            '"-Xcompiler=/Zc:__cplusplus"',
            1,
        )
    return text


def _rewrite_unix_only_args_for_msvc(text: str) -> str:
    """Turn unix-only cxx/nvcc lists (o-voxel) into MSVC-friendly Windows flags."""
    if "/std:c++" in text:
        return text
    text = re.sub(
        r'"cxx":\s*\[\s*"-O3",\s*"-std=c\+\+20"\s*\]',
        '"cxx": ["/O2", "/std:c++20", "/EHsc", "/Zc:__cplusplus", "/Zc:preprocessor"]',
        text,
    )
    text = re.sub(
        r'"nvcc":\s*\[\s*"-O3",\s*"-std=c\+\+20"\s*\]\s*\+\s*cc_flag',
        (
            '"nvcc": ["-O3", "-std=c++20", "-Xcompiler=/std:c++20", '
            '"-Xcompiler=/EHsc", "-Xcompiler=/Zc:__cplusplus", '
            '"-Xcompiler=/Zc:preprocessor", "-allow-unsupported-compiler"] + cc_flag'
        ),
        text,
    )
    text = re.sub(
        r'"nvcc":\s*\[\s*"-O3",\s*"-std=c\+\+20"\s*\]',
        (
            '"nvcc": ["-O3", "-std=c++20", "-Xcompiler=/std:c++20", '
            '"-Xcompiler=/EHsc", "-Xcompiler=/Zc:__cplusplus", '
            '"-Xcompiler=/Zc:preprocessor", "-allow-unsupported-compiler"]'
        ),
        text,
    )
    return text


def patch_windows_extension_setup(setup_py: Path) -> bool:
    """Rewrite extension setup.py: every c++17 → c++20, ensure /Zc:preprocessor.

    Idempotent only when no c++17 remains. Returns True if the file changed.
    No-op on non-Windows hosts.
    """
    if host.os_family() != "windows":
        return False
    if not setup_py.is_file():
        return False
    text = setup_py.read_text(encoding="utf-8")
    # Never skip while c++17 is still present (old early-return could).
    if (
        not setup_still_has_cxx17(text)
        and WIN_SETUP_MARKER in text
        and "/Zc:preprocessor" in text
    ):
        return False
    original = text
    text = rewrite_setup_cxx_flags(text)
    if text == original and not setup_still_has_cxx17(text):
        return False
    if WIN_SETUP_MARKER not in text:
        text = WIN_SETUP_MARKER + text
    setup_py.write_text(text, encoding="utf-8")
    print(f"Patched Windows CUDA flags in {setup_py}", flush=True)
    return True


def ensure_windows_extension_setup(setup_py: Path) -> None:
    """Patch setup.py on Windows and refuse to build if any c++17 remains.

    FlexGEMM's hardcoded /std:c++17 overrides env c++20 (cl D9025). Pip must not
    run until the file is clean.
    """
    if host.os_family() != "windows":
        return
    if not setup_py.is_file():
        raise SystemExit(f"Missing {setup_py}; cannot apply Windows CUDA compile flags.")
    patch_windows_extension_setup(setup_py)
    text = setup_py.read_text(encoding="utf-8")
    if setup_still_has_cxx17(text):
        raise SystemExit(
            f"{setup_py} still contains c++17 after patching. "
            "Refusing to build — later /std:c++17 would override C++20 (cl D9025) "
            "and break PyTorch headers. See docs/WINDOWS.md."
        )
    if "/Zc:preprocessor" not in text and "/std:c++20" in text:
        raise SystemExit(
            f"{setup_py} is missing /Zc:preprocessor (required by CCCL on "
            "CUDA 12.8+/13.x). See docs/WINDOWS.md."
        )
    print(f"Verified no c++17 in {setup_py}", flush=True)


def extension_build_env() -> dict[str, str]:
    """Full env for a pip install of a CUDA extension (PATH/CUDA_HOME + Windows flags)."""
    env = windows_cuda_build_env()
    nvcc = host.find_nvcc()
    if nvcc:
        env["PATH"] = os.pathsep.join([str(Path(nvcc).parent), env.get("PATH", "")])
        env.setdefault("CUDA_HOME", str(Path(nvcc).parent.parent))
        env.setdefault("CUDA_PATH", str(Path(nvcc).parent.parent))
    return env


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
    setup_py = target / "setup.py"
    ensure_windows_extension_setup(setup_py)
    uv = shutil.which("uv")
    run([uv, "pip", "install", "--python", str(py), "--no-build-isolation", str(target)],
        env=extension_build_env())


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
        ensure_windows_extension_setup(o_voxel / "setup.py")
        uv = shutil.which("uv")
        try:
            run([uv, "pip", "install", "--python", str(py), "--no-build-isolation",
                 str(o_voxel)], env=extension_build_env())
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"Failed to build o-voxel: {exc}") from exc


def try_flash_attn(py: Path) -> None:
    uv = shutil.which("uv")
    print("\nTrying flash-attn (optional; SDPA works if this fails)...", flush=True)
    done = subprocess.run(
        [uv, "pip", "install", "--python", str(py), "flash-attn==2.7.3",
         "--no-build-isolation"],
        env=extension_build_env(),
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
