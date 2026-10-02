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
