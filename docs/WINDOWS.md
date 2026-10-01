# Windows + NVIDIA

How to run this lab on a Windows PC with an NVIDIA graphics card.

Mac and Linux docs stay in the [README](../README.md). This page is only the Windows
route.

## What works here

| Route | In this lab on Windows? | Notes |
|---|---|---|
| **Pixal3D** | Yes | Downloads a CUDA 12 prebuilt (driver **575+**). No Visual Studio needed. |
| **Generate Image** (Qwen-Image) | Yes | Prebuilt CUDA `sd-cli`; CUDA runtime is bundled. |
| **Stable Fast 3D** | Yes | Needs **Visual Studio Build Tools** (C++) to compile the texture baker. |
| **TRELLIS.2** | Yes (CUDA) | `scripts/bootstrap_trellis_cuda.py` clones official microsoft/TRELLIS.2 into `vendor/trellis2-cuda`. Needs CUDA toolkit + VS C++ tools. Microsoft only documents Linux; Windows is best-effort. |
| **Hunyuan3D-2.1** | Yes (CUDA) | `scripts/bootstrap_hunyuan_cuda.py` clones official Tencent Hunyuan3D-2.1 into `vendor/hunyuan3d-cuda`. Officially supports Windows. |
| **Hunyuan3D-MLX (dgrauet)** | No | Apple MLX only. Prefer the CUDA Hunyuan route above. |
| **Finish** (retopo, Pixel Match, compress) | Yes | Needs **Blender 4.2+**. Repaint of sides/back is Mac-only (MLX). |

Shell scripts named `*_macos.sh` are Apple Silicon only. On Windows use the Python
bootstraps (`scripts/bootstrap_*.py`), the PowerShell wrappers, or **Setup & Status**.

## Prerequisites

1. **64-bit Windows 10 or 11**
2. **NVIDIA GPU** with a current driver: `nvidia-smi` must list the card.
   - PyTorch CUDA wheels need driver support for **CUDA 12.8+**
   - Pixal3D’s prebuilt needs **CUDA 12.9+** (driver **575** or newer)
3. **git** — [git-scm.com/download/win](https://git-scm.com/download/win)
4. **Blender 4.2+** (for Finish) — [blender.org/download](https://www.blender.org/download/)
5. **For SF3D, TRELLIS.2, or Hunyuan3D:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   with the **Desktop development with C++** workload, plus a **CUDA Toolkit** whose
   major version matches the PyTorch build (`python -m image_to_3dlab.host torch-index`).
6. **Not needed:** Homebrew, Xcode, Metal, or MLX.

## Install (one line)

In PowerShell:

```powershell
irm https://raw.githubusercontent.com/Bingeljell/image-to-3dlab/main/install.ps1 | iex
```

That clones (or updates) into `%USERPROFILE%\image-to-3dlab`, installs Python 3.11 via
`uv`, and installs **PyTorch with CUDA** on an NVIDIA machine. It downloads **no model
weights**.

Options:

```powershell
$env:I3D_DIR = "D:\lab"          # install folder
$env:I3D_REF = "main"            # branch or tag
$env:I3D_REPO = "https://github.com/YOU/image-to-3dlab.git"  # a fork
$env:I3D_YES = "1"               # no prompts (scripts / agents)
irm https://raw.githubusercontent.com/Bingeljell/image-to-3dlab/main/install.ps1 | iex
```

### From a fresh clone (no one-line installer)

```powershell
git clone https://github.com/Bingeljell/image-to-3dlab.git
cd image-to-3dlab
uv venv --python 3.11 .venv
.venv\Scripts\python.exe -m image_to_3dlab.host torch-index
# If that printed a URL (e.g. https://download.pytorch.org/whl/cu128), install CUDA torch:
uv pip install --python .venv\Scripts\python.exe --index-url <that-url> torch torchvision
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## Start the viewer

```powershell
cd $env:USERPROFILE\image-to-3dlab   # or your I3D_DIR
.venv\Scripts\python.exe viewer\serve.py
```

Open the URL it prints. Go to **Setup & Status**, pick a backend, read the size and
licence, then confirm. Weights only download after you say yes.

## Setting up TRELLIS.2 and Hunyuan3D in the lab

These are first-class Generate 3D backends on Windows/Linux NVIDIA — same dropdown and
CLI pattern as on Mac, different vendor tree under the hood.

### TRELLIS.2 (CUDA)

```powershell
.venv\Scripts\python.exe scripts\bootstrap_trellis_cuda.py
# or: .\scripts\bootstrap_trellis_cuda.ps1
```

What it does: clones `microsoft/TRELLIS.2` → `vendor/trellis2-cuda/`, installs CUDA
PyTorch + deps, compiles nvdiffrast / cumesh / flexgemm / o-voxel. **No weights yet.**

On Windows the bootstrap **sets the MSVC/CUDA compile flags for you** — you should not
need to hand-set `CXXFLAGS` / `NVCC_FLAGS`. See [CUDA extension builds](#cuda-extension-builds-trellis2)
below if a compile still fails.

Then in the viewer pick **TRELLIS.2** → Generate. First run downloads ~14 GB
(`microsoft/TRELLIS.2-4B`). Before that: request gated DINOv3 access and run
`hf auth login`.

CLI:

```powershell
vendor\trellis2-cuda\.venv\Scripts\python.exe scripts\trellis_cuda_generate.py `
  input.png output\out.glb --resolution 1024 --seed 0
```

### Hunyuan3D-2.1 (CUDA)

```powershell
.venv\Scripts\python.exe scripts\bootstrap_hunyuan_cuda.py
# or: .\scripts\bootstrap_hunyuan_cuda.ps1
```

What it does: clones `Tencent-Hunyuan/Hunyuan3D-2.1` → `vendor/hunyuan3d-cuda/`, builds
the paint rasterizer, fetches ~10 GB weights (asks first unless `--yes`).

**Licence:** Tencent Community License — not for EU / UK / South Korea.

CLI:

```powershell
vendor\hunyuan3d-cuda\.venv\Scripts\python.exe scripts\hunyuan_cuda_generate.py `
  input.png output\out.glb --model 2.1
```

In the viewer this is the **Hunyuan3D (shape + paint)** dropdown entry (same id as the
Mac MLX Xiong route; the lab picks CUDA automatically on NVIDIA).

## CLI (quick routes)

```powershell
# Pixal3D
.venv\Scripts\python.exe scripts\bootstrap_pixal3d.py
.venv\Scripts\python.exe scripts\pixal3d_generate.py input.png output.glb --seed 42

# Generate Image
.venv\Scripts\python.exe scripts\bootstrap_qwen_image.py

# Stable Fast 3D
.venv\Scripts\python.exe scripts\bootstrap_sf3d.py

# Finish (needs Blender)
.venv\Scripts\python.exe scripts\retopo_repaint.py generated.glb source.png finished.glb --faces 40000 --skip-paint
```

If Blender is installed somewhere unusual:

```powershell
$env:I2L_BLENDER = "C:\Path\To\blender.exe"
```

## CUDA extension builds (TRELLIS.2)

TRELLIS.2 compiles several CUDA packages (**nvdiffrast, nvdiffrec, CuMesh, FlexGEMM,
o-voxel** — every `setup.py` the bootstrap installs). On Windows that needs:

1. **CUDA Toolkit 12.8+** (13.x is fine) with `nvcc` on PATH or under the usual
   `NVIDIA GPU Computing Toolkit\CUDA\v*` folder.
2. **Visual Studio 2022** (17) or **Visual Studio 18** with the **Desktop development
   with C++** workload (MSVC 14.3x / 14.5x). Open the **x64 Native Tools** prompt, or
   run from a shell where `cl` works, before bootstrapping.
3. A driver new enough for that toolkit (and for your GPU — **Blackwell / sm_120**
   needs a recent driver plus a toolkit that knows `sm_120`).

`scripts/bootstrap_trellis_cuda.py` forces **only C++20** on the compile line:

| Step | What it does |
|---|---|
| Env | `DISTUTILS_USE_SDK=1`, `CL`/`CXXFLAGS` = `/Zc:preprocessor /Zc:__cplusplus` only (no `/std:` in env) |
| `setup.py` rewrite | Every extension: `c++17` → `c++20`; inject MSVC CCCL flags into known shapes (incl. nvdiffrast’s warning-only Windows list) |
| Runtime hook | **Same hook for every package**: appends `/std:c++20` + `/Zc:preprocessor` to `cxx`/`nvcc`, sanitizes ninja/spawn so 17 and 20 never mix |
| Torch patch | Rewrites `c++17` → `c++20` inside this venv’s `torch/utils/cpp_extension.py` (older wheels hardcode `-std=c++17` into ninja) |
| o-voxel source | `scripts/patch_ovoxel_msvc_narrowing.py` casts `size_t` torch shapes to `int64_t` (MSVC C2398) |
| Clean | Deletes `build/`, `*.egg-info`, `build.ninja` before `pip install` |

The post-patch check requires the CCCL **ensure** hook (`_i2l_ensure_msvc_cccl_flags`), not
merely the string `/std:c++20` somewhere in the file — an older check mistook hook
source literals for real compile flags and aborted nvdiffrast.

**Gotcha (cl D9025):** several places inject `/std:` — FlexGEMM `setup.py`, older PyTorch
ninja rules, and `CL` under nvcc `--use-local-env`. If **both** `/std:c++20` and
`/std:c++17` appear, MSVC warns `overriding …` back and forth and the final standard
is wrong. Do **not** hand-set `CL=/std:c++20` while debugging; let the bootstrap own it.
Clear stale builds: delete `vendor\trellis2-cuda\.i2l-build\FlexGEMM` (or at least its
`build\` folder / `build.ninja`) and re-run.

**Blackwell (sm_120):** if nvcc builds for the wrong arch, set
`TORCH_CUDA_ARCH_LIST=12.0` in the same shell before re-running the bootstrap.

## Common snags

| Symptom | Fix |
|---|---|
| Installer says no NVIDIA GPU | Install/update the NVIDIA driver; reopen PowerShell; `nvidia-smi` must work. |
| PyTorch stays on CPU | Re-run the installer, or install from the URL `python -m image_to_3dlab.host torch-index` prints. |
| Pixal3D refuses the prebuilt | Update the driver to **575+**. |
| TRELLIS/Hunyuan/SF3D compile fails | Install VS Build Tools (C++) + a CUDA toolkit matching PyTorch; reopen the “x64 Native Tools” shell and re-run the bootstrap. |
| `D9025` flipping `/std:c++20` ↔ `/std:c++17` on FlexGEMM | Torch and/or `setup.py` still emit C++17, or a stale `build\` ninja file. Pull latest, delete `vendor\trellis2-cuda\.i2l-build\FlexGEMM`, unset any hand-set `CL`/`CXXFLAGS` `/std:`, re-run bootstrap. See [above](#cuda-extension-builds-trellis2). |
| Bootstrap says nvdiffrast is missing `/Zc:preprocessor` after “Patched…” | Fixed: pull latest (CCCL ensure hook). Delete `vendor\trellis2-cuda\.i2l-build\nvdiffrast` and re-run. |
| o-voxel MSVC `C2398` narrowing (`size_t` → `int64_t`) | Pull latest; bootstrap runs `patch_ovoxel_msvc_narrowing.py`. Wipe `vendor\trellis2-cuda\o-voxel\build` and re-run. |
| TRELLIS fails only on Windows | Microsoft tests Linux; try the same bootstrap on Linux NVIDIA, or WSL2 with GPU. |
| Generate Image is very slow / CPU | `nvidia-smi` must see the card; update the driver and reopen the terminal. |
| Finish cannot find Blender | Install from blender.org, or set `I2L_BLENDER`. |

## What we do not do for you

- We never download model weights until you confirm in Setup & Status (or pass `--yes`).
- We never install GPU drivers, the CUDA Toolkit, Visual Studio, or Blender.

Windows support is real but still lightly tested. If something breaks, please say so in
Discussions or an issue — include your GPU, driver version (`nvidia-smi`), and the exact
command.
