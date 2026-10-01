"""Windows + NVIDIA docs and installer companions stay honest."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_windows_guide_exists_and_names_the_working_routes():
    text = (REPO / "docs" / "WINDOWS.md").read_text()
    for needle in ("Pixal3D", "Stable Fast 3D", "TRELLIS.2", "Hunyuan3D",
                   "Visual Studio Build Tools", "install.ps1", "575",
                   "microsoft/TRELLIS.2", "Hunyuan3D-2.1"):
        assert needle in text, needle


def test_readme_points_at_the_windows_guide():
    readme = (REPO / "README.md").read_text()
    assert "docs/WINDOWS.md" in readme
    assert "Linux/Windows with an NVIDIA card" in readme


def test_install_ps1_points_at_the_windows_guide():
    ps1 = (REPO / "install.ps1").read_text()
    assert "docs\\WINDOWS.md" in ps1 or "docs/WINDOWS.md" in ps1
    assert "LIMITED TESTING" in ps1


def test_powershell_bootstrap_wrappers_call_the_python_scripts():
    for name, script in (("bootstrap_pixal3d.ps1", "bootstrap_pixal3d.py"),
                         ("bootstrap_sf3d.ps1", "bootstrap_sf3d.py")):
        text = (REPO / "scripts" / name).read_text()
        assert script in text
        assert "install.ps1" in text
