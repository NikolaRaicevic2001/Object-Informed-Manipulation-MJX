"""`integrate_stream` / `window_split`: the arithmetic behind the hardware
loop's predicted start state, checked against closed forms."""

from __future__ import annotations

import numpy as np

from oim.worlds.real3d.command_stream import integrate_stream, window_split


def test_single_segment():
    t = np.array([0.0, 1.0])
    v = np.array([[2.0, -1.0], [0.0, 0.0]])
    np.testing.assert_allclose(integrate_stream(t, v, 0.25, 0.75), [1.0, -0.5])


def test_spans_segments_and_holds_last():
    t = np.array([0.0, 0.02, 0.04])
    v = np.array([[1.0], [2.0], [3.0]])
    # 0.01..0.02 at 1, 0.02..0.04 at 2, 0.04..0.10 at 3 (held)
    np.testing.assert_allclose(integrate_stream(t, v, 0.01, 0.10), [0.01 + 0.04 + 0.18])


def test_nothing_before_first_sample_or_in_empty_window():
    t = np.array([1.0])
    v = np.array([[5.0]])
    np.testing.assert_allclose(integrate_stream(t, v, 0.0, 0.5), [0.0])
    np.testing.assert_allclose(integrate_stream(t, v, 0.5, 1.5), [2.5])
    np.testing.assert_allclose(integrate_stream(t, v, 2.0, 1.0), [0.0])
    np.testing.assert_allclose(integrate_stream(np.zeros(0), np.zeros((0, 1)), 0.0, 1.0), [0.0])


def test_matches_dense_riemann_sum():
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 2, 40))
    v = rng.normal(size=(40, 5))
    t0, t1 = 0.3, 1.7
    grid = np.linspace(t0, t1, 200001)
    idx = np.searchsorted(t, grid, side="right") - 1
    ok = idx >= 0
    dense = (v[idx[ok]] * (grid[1] - grid[0])).sum(axis=0)
    np.testing.assert_allclose(integrate_stream(t, v, t0, t1), dense, atol=2e-3)


def test_window_split_geometry():
    past, future = window_split(t_state=10.0, t_now=10.005, t_publish=10.35, delay=0.05)
    assert past == (9.95, 10.005)
    assert future == (10.005, 10.35)
    # A late solve (publish already behind now) leaves no future window.
    past, future = window_split(10.0, 10.5, 10.4, 0.05)
    assert future == (10.5, 10.5)
