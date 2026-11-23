"""Map extent and geofence behaviour (#53).

There is no built-in map size: each search is bounded by a box around the start sized from the
planning horizon (or, with no horizon, the goal distance). ``map_bounds`` is an optional
geofence, off by default, respected by both A* and the optimiser when set.
"""

import json
import os
import contextlib

import numpy as np
import pytest

from coast_planner import FrontEndConfig, SparseVoxelGrid, TrajectoryPlanner


@contextlib.contextmanager
def quiet():
    """The planner prints a progress banner; keep test output readable."""
    devnull, saved = os.open(os.devnull, os.O_WRONLY), os.dup(1)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def plan(goal, **cfg):
    kw = dict(voxel_resolution=0.5, drone_physical_radius=0.49, min_altitude=0.59,
              planning_horizon=0.0)
    kw.update(cfg)
    with quiet():
        return TrajectoryPlanner(FrontEndConfig(**kw), SparseVoxelGrid(0.5, 0.49)).plan_mission(
            [0.0, 0.0, -3.0], goal)


def samples(result, n=2000):
    tr = result.sampler()
    return tr.sample(np.linspace(tr.start_time, tr.end_time, n))[:, :3]


def test_map_bounds_off_by_default():
    assert list(FrontEndConfig().map_bounds) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("distance", [52.0, 80.0, 120.0])
def test_long_goal_reached_without_map_or_horizon(distance):
    # Previously: a goal past ~50 m from the origin failed (the default map's +/-50 m planes),
    # and with no horizon a goal past 60 m stopped short (a fixed 60 m search box).
    r = plan([distance, 0.0, -3.0])
    assert r.success and r.reached_goal
    assert np.linalg.norm(samples(r)[-1] - [distance, 0.0, -3.0]) < 0.1


def test_explicit_geofence_is_respected():
    # 100 m in x and y => |x|, |y| <= 50 m; z unset => no vertical limit from map_bounds.
    r = plan([80.0, 0.0, -3.0], map_bounds=[100.0, 100.0, 0.0])
    if r.success:
        assert np.max(np.abs(samples(r)[:, :2])) <= 50.0 + 1e-2
        assert not r.reached_goal  # the goal is outside the geofence


def test_geofence_inside_reach_still_plans():
    r = plan([30.0, 5.0, -3.0], map_bounds=[100.0, 100.0, 0.0])
    assert r.success and r.reached_goal


def _demo(tmp_path, env):
    from coast_planner._demo import main
    path = tmp_path / "env.json"
    path.write_text(json.dumps(env))
    with pytest.raises(SystemExit) as e:
        main(["--env", str(path), "--mode", "sfc", "--save", str(tmp_path / "out")])
    return e.value.code


def test_demo_reports_goal_outside_geofence_as_invalid(tmp_path, capsys):
    pytest.importorskip("matplotlib")  # the demo checks for it before planning
    code = _demo(tmp_path, {"start": [0, 0, -3], "goal": [80, 0, -3], "obstacles": [],
                            "config": {"map_bounds": [100, 100, 0]}})
    assert code == 2
    assert "outside the map_bounds geofence" in capsys.readouterr().out


def test_demo_reports_goal_above_ceiling_as_invalid(tmp_path, capsys):
    pytest.importorskip("matplotlib")  # the demo checks for it before planning
    code = _demo(tmp_path, {"start": [0, 0, -3], "goal": [10, 0, -9], "obstacles": [],
                            "config": {"max_altitude": 6.0}})
    assert code == 2
    assert "above max_altitude" in capsys.readouterr().out


def test_demo_invalid_file_exits_2(tmp_path):
    assert _demo(tmp_path, {"start": [0, 0, -2]}) == 2  # no goal
