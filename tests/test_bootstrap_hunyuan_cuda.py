"""Hunyuan CUDA bootstrap: announce, licence caveat, refuse wrong hosts."""

from __future__ import annotations

import io

import bootstrap_hunyuan_cuda as boot
import pytest


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
