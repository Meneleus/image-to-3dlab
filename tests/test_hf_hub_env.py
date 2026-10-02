"""Windows HF hub must not require symlink privilege (WinError 1314)."""

from __future__ import annotations

import os

from image_to_3dlab import hf_hub_env as hub


def test_apply_sets_disable_symlinks_on_windows(monkeypatch):
    monkeypatch.setattr(hub.host, "os_family", lambda: "windows")
    env = hub.apply_windows_hf_hub_env({"PATH": "C:\\x"})
    assert env[hub.HF_HUB_DISABLE_SYMLINKS] == "1"
    assert env[hub.HF_HUB_DISABLE_SYMLINKS_WARNING] == "1"
    assert env["PATH"] == "C:\\x"


def test_apply_preserves_explicit_user_value(monkeypatch):
    monkeypatch.setattr(hub.host, "os_family", lambda: "windows")
    env = hub.apply_windows_hf_hub_env({hub.HF_HUB_DISABLE_SYMLINKS: "0"})
    assert env[hub.HF_HUB_DISABLE_SYMLINKS] == "0"


def test_apply_noop_off_windows(monkeypatch):
    monkeypatch.setattr(hub.host, "os_family", lambda: "linux")
    base = {"PATH": "/usr/bin"}
    assert hub.apply_windows_hf_hub_env(base) == base
    assert hub.HF_HUB_DISABLE_SYMLINKS not in hub.apply_windows_hf_hub_env(base)


def test_ensure_process_sets_os_environ(monkeypatch):
    monkeypatch.setattr(hub.host, "os_family", lambda: "windows")
    monkeypatch.delenv(hub.HF_HUB_DISABLE_SYMLINKS, raising=False)
    monkeypatch.delenv(hub.HF_HUB_DISABLE_SYMLINKS_WARNING, raising=False)
    hub.ensure_process_windows_hf_hub_env()
    assert os.environ[hub.HF_HUB_DISABLE_SYMLINKS] == "1"
