"""The demo's workaround for Python 3.13.0 venvs on Windows (Tcl can't find its library).

Runs on any OS by faking ``sys.platform`` and a base-Python layout with its Tcl/Tk folders.
"""

import sys

from coast_planner import _demo


def _fake_base_python(tmp_path):
    (tmp_path / "tcl" / "tcl8.6").mkdir(parents=True)
    (tmp_path / "tcl" / "tcl8.6" / "init.tcl").write_text("")
    (tmp_path / "tcl" / "tk8.6").mkdir(parents=True)
    (tmp_path / "tcl" / "tk8.6" / "tk.tcl").write_text("")
    return tmp_path


def test_points_tcl_at_base_python_on_windows(tmp_path, monkeypatch):
    base = _fake_base_python(tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.delenv("TCL_LIBRARY", raising=False)
    monkeypatch.delenv("TK_LIBRARY", raising=False)
    assert _demo._point_tcl_at_base_python()
    import os
    assert os.environ["TCL_LIBRARY"] == str(base / "tcl" / "tcl8.6")
    assert os.environ["TK_LIBRARY"] == str(base / "tcl" / "tk8.6")


def test_leaves_existing_tcl_library_alone(tmp_path, monkeypatch):
    base = _fake_base_python(tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "base_prefix", str(base))
    monkeypatch.setenv("TCL_LIBRARY", "/somewhere/else")   # e.g. set by conda
    assert not _demo._point_tcl_at_base_python()
    import os
    assert os.environ["TCL_LIBRARY"] == "/somewhere/else"


def test_does_nothing_off_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("TCL_LIBRARY", raising=False)
    assert not _demo._point_tcl_at_base_python()
