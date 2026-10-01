"""TRELLIS CUDA bootstrap: announce, refuse wrong hosts, never fetch without --yes."""

from __future__ import annotations

import io

import bootstrap_trellis_cuda as boot
import pytest


FLEXGEMM_SETUP = '''\
from setuptools import setup
import platform

if platform.system() == "Windows":
    extra_compile_args = {
        "cxx": ["/O2", "/std:c++17", "/EHsc", "/openmp", "/permissive-", "/Zc:__cplusplus"],
        "nvcc": ["-O3", "-std=c++17", "-Xcompiler=/std:c++17", "-Xcompiler=/EHsc",
                 "-Xcompiler=/permissive-", "-Xcompiler=/Zc:__cplusplus"],
    }
else:
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
        "nvcc": ["-O3", "-std=c++17"],
    }
'''

O_VOXEL_SETUP = '''\
extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3","-std=c++17"] + cc_flag,
            }
'''


def test_announcement_names_backend_route_and_weight_policy():
    text = boot.announcement()
    assert "TRELLIS.2" in text and "CUDA" in text
    assert "14 GB" in text
    assert "No model weights" in text or "no model weights" in text.lower() or "not fetched" in text.lower() or "on first" in text


def test_announcement_mentions_windows_compile_flags(monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    text = boot.announcement()
    assert "Zc:preprocessor" in text or "C++20" in text


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


def test_windows_cuda_build_env_sets_msvc_flags(monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    env = boot.windows_cuda_build_env({"PATH": "C:\\x"})
    assert env["DISTUTILS_USE_SDK"] == "1"
    assert "/std:c++20" in env["CXXFLAGS"]
    assert "/Zc:preprocessor" in env["CXXFLAGS"]
    assert "/Zc:__cplusplus" in env["CXXFLAGS"]
    assert "/std:c++20" in env["CL"]
    assert "-std=c++20" in env["NVCC_FLAGS"]
    assert "-Xcompiler=/Zc:preprocessor" in env["NVCC_FLAGS"]
    assert env["NVCC_PREPEND_FLAGS"] == env["NVCC_FLAGS"]


def test_windows_cuda_build_env_noop_on_linux(monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda: "linux")
    base = {"PATH": "/usr/bin"}
    assert boot.windows_cuda_build_env(base) == base
    assert "DISTUTILS_USE_SDK" not in boot.windows_cuda_build_env(base)


def test_patch_windows_extension_setup_bumps_flexgemm_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    setup = tmp_path / "setup.py"
    setup.write_text(FLEXGEMM_SETUP, encoding="utf-8")
    assert boot.patch_windows_extension_setup(setup) is True
    text = setup.read_text(encoding="utf-8")
    assert boot.WIN_SETUP_MARKER in text
    assert "/std:c++20" in text
    assert "/std:c++17" not in text
    assert "-std=c++20" in text
    assert "-Xcompiler=/std:c++20" in text
    assert "/Zc:preprocessor" in text
    assert "-Xcompiler=/Zc:preprocessor" in text
    # Idempotent
    assert boot.patch_windows_extension_setup(setup) is False


def test_patch_windows_extension_setup_rewrites_ovoxel_unix_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    setup = tmp_path / "setup.py"
    setup.write_text(O_VOXEL_SETUP, encoding="utf-8")
    assert boot.patch_windows_extension_setup(setup) is True
    text = setup.read_text(encoding="utf-8")
    assert '"/std:c++20"' in text
    assert "/Zc:preprocessor" in text
    assert "-Xcompiler=/Zc:preprocessor" in text
    assert '"-std=c++17"' not in text


def test_patch_windows_extension_setup_noop_off_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "linux")
    setup = tmp_path / "setup.py"
    setup.write_text(FLEXGEMM_SETUP, encoding="utf-8")
    assert boot.patch_windows_extension_setup(setup) is False
    assert setup.read_text(encoding="utf-8") == FLEXGEMM_SETUP
