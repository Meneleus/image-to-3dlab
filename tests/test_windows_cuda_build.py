"""Shared Windows CUDA build env helpers used by TRELLIS and Hunyuan bootstraps."""

from __future__ import annotations

from image_to_3dlab import windows_cuda_build as wcb


def test_windows_cuda_build_env_sets_sdk_and_zc(monkeypatch):
    monkeypatch.setattr(wcb.host, "os_family", lambda: "windows")
    env = wcb.windows_cuda_build_env({"PATH": "C:\\x", "CL": "/std:c++17"})
    assert env["DISTUTILS_USE_SDK"] == "1"
    assert env["CL"] == wcb.WIN_CL_FLAGS
    assert "/std:" not in env["CL"]
    assert "/Zc:preprocessor" in env["CL"]
    assert env["NVCC_PREPEND_FLAGS"] == wcb.WIN_NVCC_FLAGS


def test_windows_cuda_build_env_noop_off_windows(monkeypatch):
    monkeypatch.setattr(wcb.host, "os_family", lambda: "linux")
    base = {"PATH": "/usr/bin"}
    assert wcb.windows_cuda_build_env(base) == base


def test_extension_build_env_adds_cuda_home(monkeypatch, tmp_path):
    monkeypatch.setattr(wcb.host, "os_family", lambda: "windows")
    nvcc = tmp_path / "bin" / "nvcc.exe"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("", encoding="utf-8")
    monkeypatch.setattr(wcb.host, "find_nvcc", lambda: str(nvcc))
    env = wcb.extension_build_env({"PATH": "C:\\old"})
    assert env["DISTUTILS_USE_SDK"] == "1"
    assert str(nvcc.parent) in env["PATH"]
    assert env["CUDA_HOME"] == str(tmp_path)
