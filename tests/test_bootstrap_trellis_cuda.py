"""TRELLIS CUDA bootstrap: announce, refuse wrong hosts, never fetch without --yes."""

from __future__ import annotations

import io

import bootstrap_trellis_cuda as boot
import pytest


def test_announcement_names_backend_route_and_weight_policy():
    text = boot.announcement()
    assert "TRELLIS.2" in text and "CUDA" in text
    assert "14 GB" in text
    assert "No model weights" in text or "no model weights" in text.lower() or "not fetched" in text.lower() or "on first" in text


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
