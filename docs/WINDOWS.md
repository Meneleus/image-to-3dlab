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
| **Finish** (retopo, Pixel Match, compress) | Yes | Needs **Blender 4.2+**. Repaint of sides/back is Mac-only (MLX). |
| **TRELLIS.2** | No (Mac port only) | Use [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) on CUDA. |
| **Hunyuan3D-MLX** | No (Apple MLX only) | Use [Tencent Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) on CUDA. |

Shell scripts named `*_macos.sh` are Apple Silicon only. On Windows use the Python
bootstraps (`scripts/bootstrap_*.py`) or the viewer’s **Setup & Status** page.

## Prerequisites

1. **64-bit Windows 10 or 11**
2. **NVIDIA GPU** with a current driver: `nvidia-smi` must list the card.
   - PyTorch CUDA wheels need driver support for **CUDA 12.8+**
   - Pixal3D’s prebuilt needs **CUDA 12.9+** (driver **575** or newer)
3. **git** — [git-scm.com/download/win](https://git-scm.com/download/win)
4. **Blender 4.2+** (for Finish) — [blender.org/download](https://www.blender.org/download/)
5. **Optional, for SF3D only:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   with the **Desktop development with C++** workload. Not needed for Pixal3D or
   Generate Image.
6. **Optional:** a CUDA Toolkit matching your PyTorch major version, if you want SF3D’s
   CUDA texture baker instead of its CPU baker. The model still runs on the GPU either way.

You do **not** need Homebrew, Xcode, Metal, or MLX.

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

## CLI (same engines)

```powershell
# Pixal3D
.venv\Scripts\python.exe scripts\bootstrap_pixal3d.py
.venv\Scripts\python.exe scripts\pixal3d_generate.py input.png output.glb --seed 42

# Generate Image weights + binary
.venv\Scripts\python.exe scripts\bootstrap_qwen_image.py

# Stable Fast 3D (needs VS Build Tools for the C++ extensions)
.venv\Scripts\python.exe scripts\bootstrap_sf3d.py

# Finish (needs Blender on PATH or in Program Files)
.venv\Scripts\python.exe scripts\retopo_repaint.py generated.glb source.png finished.glb --faces 40000 --skip-paint
```

If Blender is installed somewhere unusual:

```powershell
$env:I2L_BLENDER = "C:\Path\To\blender.exe"
```

## Mac-only routes: what to use instead

**TRELLIS.2** and **Hunyuan3D** in this repo are Apple Silicon ports (Metal / MLX). Setup
& Status will say they are not available on this machine and point at the official NVIDIA
repos. That is intentional: downloading Mac-only weights here would waste disk and then
fail.

| Want | On Windows use |
|---|---|
| TRELLIS.2 | [microsoft/TRELLIS.2](https://github.com/microsoft/TRELLIS.2) |
| Hunyuan3D | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) |
| Fast textured mesh in *this* lab | **Pixal3D** (recommended start) |
| Quick lower-fidelity mesh | **Stable Fast 3D** |

## Common snags

| Symptom | Fix |
|---|---|
| Installer says no NVIDIA GPU | Install/update the NVIDIA driver; reopen PowerShell; `nvidia-smi` must work. |
| PyTorch stays on CPU | Re-run the installer (it replaces a CPU-only torch), or install from the URL `python -m image_to_3dlab.host torch-index` prints. Driver must support CUDA 12.8+. |
| Pixal3D refuses the prebuilt | Update the driver to **575+** (CUDA 12.9). |
| SF3D compile fails | Install Visual Studio Build Tools with C++, then run `bootstrap_sf3d.py` again. |
| Generate Image is very slow / CPU | `nvidia-smi` must see the card; update the driver and reopen the terminal. |
| Finish cannot find Blender | Install from blender.org, or set `I2L_BLENDER`. |
| Paths with spaces | Quote them: `cd "D:\My Lab"`. |

## What we do not do for you

- We never download model weights until you confirm in Setup & Status (or pass `--yes` on
  a bootstrap).
- We never install GPU drivers, the CUDA Toolkit, Visual Studio, or Blender.
- We do not ship TRELLIS/Hunyuan CUDA builds inside this lab yet; use the upstream repos
  linked above.

Windows support is real but still lightly tested. If something breaks, please say so in
Discussions or an issue — include your GPU, driver version (`nvidia-smi`), and the exact
command.
