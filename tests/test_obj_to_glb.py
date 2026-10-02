"""OBJ→GLB without pip bpy: blender.exe first, then bpy, then trimesh."""

from __future__ import annotations

from types import SimpleNamespace

from image_to_3dlab import obj_to_glb as o2g


def test_convert_via_blender_exe_noop_without_blender(monkeypatch, tmp_path):
    monkeypatch.setattr(o2g, "find_blender", lambda: None)
    assert o2g.convert_via_blender_exe(tmp_path / "a.obj", tmp_path / "a.glb") is False


def test_convert_via_blender_exe_invokes_find_blender(monkeypatch, tmp_path):
    exe = tmp_path / "blender.exe"
    exe.write_text("")
    obj = tmp_path / "mesh.obj"
    obj.write_text("v 0 0 0\n")
    glb = tmp_path / "mesh.glb"
    seen: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        glb.write_bytes(b"glTF")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(o2g, "find_blender", lambda: exe)
    assert o2g.convert_via_blender_exe(obj, glb, run=fake_run) is True
    assert seen and seen[0][0] == str(exe)
    assert "--background" in seen[0] and "--python" in seen[0]
    assert str(obj.resolve()) in seen[0]
    assert str(glb.resolve()) in seen[0]


def test_convert_obj_to_glb_tries_routes_in_order(monkeypatch, tmp_path):
    order: list[str] = []
    monkeypatch.setattr(
        o2g, "convert_via_blender_exe",
        lambda *a, **k: order.append("blender") or False,
    )
    monkeypatch.setattr(
        o2g, "convert_via_bpy",
        lambda *a, **k: order.append("bpy") or False,
    )
    monkeypatch.setattr(
        o2g, "convert_via_trimesh",
        lambda *a, **k: order.append("trimesh") or True,
    )
    assert o2g.convert_obj_to_glb(tmp_path / "a.obj", tmp_path / "a.glb") is True
    assert order == ["blender", "bpy", "trimesh"]


def test_convert_via_bpy_false_without_module(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "bpy":
            raise ImportError("no bpy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", boom)
    assert o2g.convert_via_bpy("a.obj", "a.glb") is False
