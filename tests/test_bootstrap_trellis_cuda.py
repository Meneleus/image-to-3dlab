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

NVDIFFRAST_SETUP = '''\
import setuptools
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setuptools.setup(
    ext_modules=[
        CUDAExtension(
            "_nvdiffrast_c",
            sources=["csrc/common/common.cpp"],
            extra_compile_args={
                "cxx": ["-DNVDR_TORCH"]
                + (["/wd4067", "/wd4624", "/wd4996"] if os.name == "nt" else []),
                "nvcc": ["-DNVDR_TORCH", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
'''

NVDIFFREC_SETUP = '''\
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

c_flags = ['-DNVDR_TORCH']
nvcc_flags = ['-DNVDR_TORCH']

setup(
    name='nvdiffrec_render',
    ext_modules=[
        CUDAExtension(
            name='nvdiffrec_render.renderutils._C',
            sources=['x.cpp'],
            extra_compile_args={'cxx': c_flags, 'nvcc': nvcc_flags},
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
'''

# Every package the bootstrap installs on Windows (CuMesh matches FlexGEMM's shape).
ALL_SETUP_FIXTURES = {
    "FlexGEMM": FLEXGEMM_SETUP,
    "CuMesh": FLEXGEMM_SETUP,
    "o-voxel": O_VOXEL_SETUP,
    "nvdiffrast": NVDIFFRAST_SETUP,
    "nvdiffrec": NVDIFFREC_SETUP,
}


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


def test_windows_cuda_build_env_sets_zc_not_std(monkeypatch):
    """CL must not carry /std:c++20 — that races with torch under --use-local-env."""
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    env = boot.windows_cuda_build_env({"PATH": "C:\\x", "CL": "/std:c++17 /something"})
    assert env["DISTUTILS_USE_SDK"] == "1"
    assert "/Zc:preprocessor" in env["CXXFLAGS"]
    assert "/Zc:__cplusplus" in env["CXXFLAGS"]
    assert "/std:" not in env["CL"]
    assert "/std:" not in env["CXXFLAGS"]
    assert "-std=" not in env["NVCC_FLAGS"]
    assert "-Xcompiler=/Zc:preprocessor" in env["NVCC_FLAGS"]
    assert env["NVCC_PREPEND_FLAGS"] == env["NVCC_FLAGS"]


def test_windows_cuda_build_env_noop_on_linux(monkeypatch):
    monkeypatch.setattr(boot.host, "os_family", lambda: "linux")
    base = {"PATH": "/usr/bin"}
    assert boot.windows_cuda_build_env(base) == base
    assert "DISTUTILS_USE_SDK" not in boot.windows_cuda_build_env(base)


def test_sanitize_std_flags_keeps_only_cxx20():
    mixed = [
        "/O2", "/std:c++20", "/EHsc", "/std:c++17",
        "-std=c++17", "-Xcompiler", "/std:c++17", "-Xcompiler=/std:c++20",
    ]
    out = boot.sanitize_std_flags(mixed)
    joined = " ".join(out)
    assert "c++17" not in joined.lower()
    assert out.count("/std:c++20") == 1
    assert out.count("-std=c++20") == 1
    assert out.count("-Xcompiler=/std:c++20") == 1
    assert "/O2" in out and "/EHsc" in out


def test_ensure_msvc_cccl_flags_appends_to_sparse_nvdiffrast_args():
    sparse = {
        "cxx": ["-DNVDR_TORCH", "/wd4067"],
        "nvcc": ["-DNVDR_TORCH", "-lineinfo"],
    }
    out = boot.ensure_msvc_cccl_flags(sparse)
    for flag in boot.WIN_MSVC_CXX_FLAGS:
        assert flag in out["cxx"]
    for flag in boot.WIN_MSVC_NVCC_FLAGS:
        assert flag in out["nvcc"]


def test_rewrite_setup_cxx_flags_removes_every_cxx17():
    out = boot.rewrite_setup_cxx_flags(FLEXGEMM_SETUP)
    assert "c++17" not in out.lower()
    assert "/std:c++20" in out
    assert "-std=c++20" in out
    assert "-Xcompiler=/std:c++20" in out
    assert "/Zc:preprocessor" in out
    assert "-Xcompiler=/Zc:preprocessor" in out


@pytest.mark.parametrize("name", sorted(ALL_SETUP_FIXTURES))
def test_ensure_passes_for_every_bootstrap_extension(monkeypatch, tmp_path, name, capsys):
    """Uniform path: nvdiffrast/nvdiffrec/FlexGEMM/o-voxel all get CCCL ensure hook."""
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    setup = tmp_path / "setup.py"
    setup.write_text(ALL_SETUP_FIXTURES[name], encoding="utf-8")
    boot.ensure_windows_extension_setup(setup)
    text = setup.read_text(encoding="utf-8")
    assert boot.setup_still_has_cxx17(text) is False
    assert boot.setup_has_cccl_ensure_hook(text)
    assert boot.WIN_HOOK_ENSURE_NAME in text
    assert "/Zc:preprocessor" in text
    assert "/std:c++20" in text
    assert "Verified Windows CCCL/C++20 hook" in capsys.readouterr().out
    # Idempotent
    assert boot.patch_windows_extension_setup(setup) is False


def test_nvdiffrast_gets_source_level_cccl_flags(monkeypatch, tmp_path):
    """nvdiffrast upstream has no /std or Zc — rewrite must inject them into lists."""
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    setup = tmp_path / "setup.py"
    setup.write_text(NVDIFFRAST_SETUP, encoding="utf-8")
    boot.ensure_windows_extension_setup(setup)
    outside = boot.setup_text_outside_hook(setup.read_text(encoding="utf-8"))
    assert '"/std:c++20"' in outside
    assert '"/Zc:preprocessor"' in outside
    assert '"-Xcompiler=/Zc:preprocessor"' in outside


def test_nvdiffrec_gets_windows_cccl_block(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    setup = tmp_path / "setup.py"
    setup.write_text(NVDIFFREC_SETUP, encoding="utf-8")
    boot.ensure_windows_extension_setup(setup)
    outside = boot.setup_text_outside_hook(setup.read_text(encoding="utf-8"))
    assert "/Zc:preprocessor" in outside
    assert "os.name == 'nt'" in outside


def test_old_hook_without_cccl_ensure_is_replaced(monkeypatch, tmp_path):
    """Stale 9bfd876-era hook had /std:c++20 literals but no CCCL ensure — upgrade it."""
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    old_hook = (
        boot.WIN_HOOK_BEGIN
        + 'def _i2l_sanitize_std_flags(flags):\n    return flags\n'
        + 'msvc_std = "/std:c++20"\n'
        + boot.WIN_HOOK_END
    )
    setup = tmp_path / "setup.py"
    setup.write_text(old_hook + "\n" + NVDIFFRAST_SETUP, encoding="utf-8")
    # This was the false failure: /std in hook, no /Zc in file → old validator aborted.
    assert "/std:c++20" in setup.read_text(encoding="utf-8")
    assert boot.setup_has_cccl_ensure_hook(setup.read_text(encoding="utf-8")) is False
    boot.ensure_windows_extension_setup(setup)
    assert boot.setup_has_cccl_ensure_hook(setup.read_text(encoding="utf-8"))


def test_patch_windows_extension_setup_bumps_flexgemm_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    setup = tmp_path / "setup.py"
    setup.write_text(FLEXGEMM_SETUP, encoding="utf-8")
    assert boot.patch_windows_extension_setup(setup) is True
    text = setup.read_text(encoding="utf-8")
    assert boot.WIN_SETUP_MARKER in text
    assert boot.setup_has_cccl_ensure_hook(text)
    assert boot.setup_still_has_cxx17(text) is False
    assert boot.patch_windows_extension_setup(setup) is False


def test_patch_rewrites_even_when_marker_present_but_cxx17_remains(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    setup = tmp_path / "setup.py"
    setup.write_text(
        boot.WIN_SETUP_MARKER
        + 'flags = ["/std:c++17", "/Zc:preprocessor", "-std=c++17"]\n',
        encoding="utf-8",
    )
    assert boot.patch_windows_extension_setup(setup) is True
    text = setup.read_text(encoding="utf-8")
    assert boot.setup_still_has_cxx17(text) is False
    assert boot.setup_has_cccl_ensure_hook(text)


def test_ensure_windows_extension_setup_refuses_leftover_cxx17(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    setup = tmp_path / "setup.py"
    setup.write_text(FLEXGEMM_SETUP, encoding="utf-8")
    monkeypatch.setattr(boot, "patch_windows_extension_setup", lambda _p: False)
    with pytest.raises(SystemExit, match=r"still contains c\+\+17"):
        boot.ensure_windows_extension_setup(setup)


def test_patch_windows_extension_setup_rewrites_ovoxel_unix_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    setup = tmp_path / "setup.py"
    setup.write_text(O_VOXEL_SETUP, encoding="utf-8")
    assert boot.patch_windows_extension_setup(setup) is True
    text = setup.read_text(encoding="utf-8")
    assert boot.setup_still_has_cxx17(text) is False
    assert '"/std:c++20"' in boot.setup_text_outside_hook(text)
    assert "/Zc:preprocessor" in text


def test_patch_windows_extension_setup_noop_off_windows(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "linux")
    setup = tmp_path / "setup.py"
    setup.write_text(FLEXGEMM_SETUP, encoding="utf-8")
    assert boot.patch_windows_extension_setup(setup) is False
    assert setup.read_text(encoding="utf-8") == FLEXGEMM_SETUP
    boot.ensure_windows_extension_setup(setup)
    assert setup.read_text(encoding="utf-8") == FLEXGEMM_SETUP


def test_clean_extension_build_artifacts_removes_stale_dirs(tmp_path):
    build = tmp_path / "build"
    build.mkdir()
    (build / "build.ninja").write_text("c++17", encoding="utf-8")
    egg = tmp_path / "flex_gemm.egg-info"
    egg.mkdir()
    boot.clean_extension_build_artifacts(tmp_path)
    assert not build.exists()
    assert not egg.exists()


def test_bootstrap_extension_list_is_fully_covered():
    """Guard: every EXTENSIONS name + o-voxel has a fixture in ALL_SETUP_FIXTURES."""
    names = {name for name, _, _ in boot.EXTENSIONS} | {"o-voxel"}
    assert names <= set(ALL_SETUP_FIXTURES), names - set(ALL_SETUP_FIXTURES)


def test_patch_torch_cpp_extension_cxx20(monkeypatch, tmp_path):
    monkeypatch.setattr(boot.host, "os_family", lambda: "windows")
    fake = tmp_path / "cpp_extension.py"
    fake.write_text('cuda_cflags = ["-std=c++17"]\nflag = "/std:c++17"\n', encoding="utf-8")
    py = tmp_path / "python.exe"
    py.write_text("", encoding="utf-8")

    def fake_run(cmd, capture_output=False, text=False, check=False):
        class R:
            returncode = 0
            stdout = str(fake) + "\n"
            stderr = ""
        return R()

    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    assert boot.patch_torch_cpp_extension_cxx20(py) is True
    text = fake.read_text(encoding="utf-8")
    assert "c++17" not in text
    assert "-std=c++20" in text and "/std:c++20" in text
    assert boot.patch_torch_cpp_extension_cxx20(py) is False


def test_try_flash_attn_installs_psutil_before_flash_attn(monkeypatch, tmp_path):
    """flash-attn@2.7.3 needs psutil at build time under --no-build-isolation."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(boot.shutil, "which", lambda _n: "/usr/bin/uv")
    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    monkeypatch.setattr(boot, "extension_build_env", lambda: {"PATH": "/x"})
    boot.try_flash_attn(tmp_path / "python")

    assert len(calls) >= 2
    assert "psutil" in calls[0]
    assert any("flash-attn==2.7.3" in part for part in calls[1])
    assert "--no-build-isolation" in calls[1]
    # Must not abort the bootstrap on flash-attn failure path either.
    assert calls[0].index("psutil") > 0


def test_try_flash_attn_soft_fails_when_build_fails(monkeypatch, tmp_path, capsys):
    def fake_run(cmd, **kwargs):
        class R:
            returncode = 0 if "psutil" in cmd else 1

        return R()

    monkeypatch.setattr(boot.shutil, "which", lambda _n: "/usr/bin/uv")
    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    monkeypatch.setattr(boot, "extension_build_env", dict)
    boot.try_flash_attn(tmp_path / "python")  # must not raise
    out = capsys.readouterr().out.lower()
    assert "sdpa" in out
    assert "flash-attn not installed" in out


def test_try_flash_attn_soft_fails_when_psutil_missing(monkeypatch, tmp_path, capsys):
    def fake_run(cmd, **kwargs):
        class R:
            returncode = 1

        return R()

    monkeypatch.setattr(boot.shutil, "which", lambda _n: "/usr/bin/uv")
    monkeypatch.setattr(boot.subprocess, "run", fake_run)
    boot.try_flash_attn(tmp_path / "python")
    out = capsys.readouterr().out.lower()
    assert "psutil" in out and "sdpa" in out
