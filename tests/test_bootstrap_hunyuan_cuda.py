"""Hunyuan CUDA bootstrap: announce, licence caveat, refuse wrong hosts."""

from __future__ import annotations

import io

import bootstrap_hunyuan_cuda as boot
import pytest

from image_to_3dlab.windows_cuda_build import windows_cuda_build_env


def test_announcement_names_backend_size_and_territorial_licence():
    text = boot.announcement()
    assert "Hunyuan3D-2.1" in text and "CUDA" in text
    assert "10.0 GB" in text or "10 GB" in text
    assert "EU" in text and "UK" in text


def test_wrong_host_refuses(monkeypatch, capsys):
    monkeypatch.setattr(boot.host, "host_platform", lambda: boot.host.APPLE)
    monkeypatch.setattr(boot, "ensure_clone", lambda: pytest.fail("cloned"))
    assert boot.main(["--yes"]) == 1
    assert "Nothing downloaded" in capsys.readouterr().out


def test_no_yes_without_terminal_refuses(monkeypatch):
    monkeypatch.setattr(boot.host, "host_platform", lambda: boot.host.NVIDIA)
    monkeypatch.setattr(boot.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(boot, "ensure_clone", lambda: pytest.fail("cloned"))
    assert boot.main([]) == 1


def test_code_and_weights_halves(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "host_platform", lambda: boot.host.NVIDIA)
    called = []
    monkeypatch.setattr(boot, "ensure_clone", lambda: called.append("clone"))
    monkeypatch.setattr(boot, "ensure_venv", lambda: called.append("venv") or tmp_path / "python")
    monkeypatch.setattr(boot, "install_code", lambda py: called.append("code"))
    monkeypatch.setattr(boot, "install_weights", lambda: called.append("weights"))
    assert boot.main(["--yes", "--code-only"]) == 0
    assert boot.main(["--yes", "--weights-only"]) == 0
    assert "code" in called and "weights" in called


def test_install_cuda_extension_sets_distutils_use_sdk_and_skips_editable(
    monkeypatch, tmp_path,
):
    """Windows VC-activated shells need DISTUTILS_USE_SDK=1; prefer non-editable."""
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    target = tmp_path / "custom_rasterizer"
    target.mkdir()
    (target / "setup.py").write_text("from setuptools import setup\nsetup(name='x')\n",
                                     encoding="utf-8")
    seen: dict = {}

    monkeypatch.setattr(boot.shutil, "which", lambda _n: "/usr/bin/uv")
    monkeypatch.setattr(boot.trellis_boot, "ensure_windows_extension_setup",
                        lambda p: seen.setdefault("setup", p))
    monkeypatch.setattr(boot.trellis_boot, "clean_extension_build_artifacts",
                        lambda p: seen.setdefault("clean", p))

    def fake_run(cmd, *, cwd=None, env=None):
        seen["cmd"] = list(cmd)
        seen["env"] = dict(env or {})

    monkeypatch.setattr(boot, "run", fake_run)
    monkeypatch.setattr(
        boot, "extension_build_env",
        lambda: windows_cuda_build_env({"PATH": "C:\\x"}),
    )

    boot.install_cuda_extension(tmp_path / "python.exe", target, "custom_rasterizer")

    assert seen["setup"] == target / "setup.py"
    assert seen["clean"] == target
    assert "-e" not in seen["cmd"]
    assert "--force-reinstall" in seen["cmd"]
    assert "--no-build-isolation" in seen["cmd"]
    assert str(target) in seen["cmd"]
    assert seen["env"].get("DISTUTILS_USE_SDK") == "1"
    assert "/Zc:preprocessor" in seen["env"].get("CL", "")


def test_windows_cuda_build_env_shared_with_trellis(monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    env = windows_cuda_build_env({"PATH": "C:\\x"})
    assert env["DISTUTILS_USE_SDK"] == "1"
    assert "/std:" not in env["CL"]
    assert "/Zc:preprocessor" in env["CL"]


def test_patch_custom_rasterizer_msvc_narrowing_runs_on_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))

    monkeypatch.setattr(boot, "run", fake_run)
    raster = tmp_path / "custom_rasterizer"
    boot.patch_custom_rasterizer_msvc_narrowing(raster)
    assert len(seen) == 1
    assert "patch_hunyuan_rasterizer_msvc_narrowing.py" in seen[0][1]
    assert str(raster) in seen[0]


def test_patch_custom_rasterizer_msvc_narrowing_noop_off_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "linux")
    monkeypatch.setattr(boot, "run", lambda *a, **k: pytest.fail("should not run"))
    boot.patch_custom_rasterizer_msvc_narrowing(tmp_path)


def test_runtime_packages_prefer_onnxruntime_gpu_on_windows(monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    pkgs = boot.runtime_packages()
    assert "pymeshlab==2025.7.post1" in pkgs
    assert "realesrgan" in pkgs
    assert "onnxruntime-gpu" in pkgs and "onnxruntime" not in pkgs
    assert "msvc-runtime" in pkgs


def test_runtime_packages_use_cpu_onnxruntime_on_linux(monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda: "linux")
    pkgs = boot.runtime_packages()
    assert "onnxruntime" in pkgs and "onnxruntime-gpu" not in pkgs
    assert "msvc-runtime" not in pkgs


def test_pymeshlab_and_ort_install_into_the_vendor_venv(monkeypatch, tmp_path):
    """hy3dshape + rembg need these in vendor/hunyuan3d-cuda/.venv, not lab root."""
    assert any(p.startswith("pymeshlab") for p in boot.PIP_PACKAGES)
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    monkeypatch.setattr(boot.host, "driver_cuda_version", lambda: "13.0")
    monkeypatch.setattr(boot.host, "torch_cuda_index", lambda *_: "https://download.pytorch.org/whl/cu130")
    monkeypatch.setattr(boot.shutil, "which", lambda _n: "/usr/bin/uv")
    monkeypatch.setattr(boot.trellis_boot, "patch_torch_cpp_extension_cxx20", lambda *_: None)
    monkeypatch.setattr(boot, "patch_custom_rasterizer_msvc_narrowing", lambda *_: None)
    monkeypatch.setattr(boot, "patch_mesh_utils_blender", lambda *_: None)
    monkeypatch.setattr(boot, "install_cuda_extension", lambda *a, **k: None)
    # No rasterizer / renderer dirs in tmp — skip those compile steps.
    monkeypatch.setattr(boot, "VENDOR", tmp_path)
    monkeypatch.setattr(boot.urllib.request, "urlretrieve", lambda *a, **k: None)
    seen: list[list[str]] = []
    monkeypatch.setattr(boot, "run", lambda cmd, **kw: seen.append(list(cmd)))
    vendor_py = tmp_path / ".venv" / "Scripts" / "python.exe"
    boot.install_code(vendor_py)
    runtime = [c for c in seen if "pymeshlab==2025.7.post1" in c]
    assert runtime, seen
    assert "--python" in runtime[0] and str(vendor_py) in runtime[0]
    assert "msvc-runtime" in runtime[0]
    assert "onnxruntime-gpu" in runtime[0]
    assert "realesrgan" in runtime[0]


def test_install_code_patches_mesh_utils_for_blender(monkeypatch, tmp_path):
    diff = tmp_path / "hy3dpaint" / "DifferentiableRenderer"
    diff.mkdir(parents=True)
    (diff / "mesh_utils.py").write_text("import bpy\n", encoding="utf-8")
    (diff / "mesh_inpaint_processor.cpp").write_text("// x\n", encoding="utf-8")
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    monkeypatch.setattr(boot.host, "driver_cuda_version", lambda: "13.0")
    monkeypatch.setattr(boot.host, "torch_cuda_index", lambda *_: "https://download.pytorch.org/whl/cu130")
    monkeypatch.setattr(boot.shutil, "which", lambda _n: "/usr/bin/uv")
    monkeypatch.setattr(boot.trellis_boot, "patch_torch_cpp_extension_cxx20", lambda *_: None)
    monkeypatch.setattr(boot, "patch_custom_rasterizer_msvc_narrowing", lambda *_: None)
    monkeypatch.setattr(boot, "install_cuda_extension", lambda *a, **k: None)
    monkeypatch.setattr(boot, "VENDOR", tmp_path)
    monkeypatch.setattr(boot.urllib.request, "urlretrieve", lambda *a, **k: None)
    seen: list[list[str]] = []
    monkeypatch.setattr(boot, "run", lambda cmd, **kw: seen.append(list(cmd)))
    boot.install_code(tmp_path / "python.exe")
    assert any("patch_hunyuan_mesh_utils_blender.py" in str(c) for c in seen)
