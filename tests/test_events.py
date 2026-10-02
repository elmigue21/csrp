import numpy as np

from conftest import FIX_S, GAZE_HZ, SACC_DEG, gaze_path, unit_from_angles
from events import angular_speed_deg_s, detect_events

HZ = 240.0


def _grid_xyz(seconds=6.0, seed=0):
    t, az, el = gaze_path(seconds, np.random.default_rng(seed))
    grid = np.arange(t[0], t[-1], 1 / HZ)
    xyz = unit_from_angles(np.interp(grid, t, az), np.interp(grid, t, el))
    return xyz


def test_speed_of_constant_rotation():
    deg_per_s = 100.0
    az = np.arange(0, 1, 1 / HZ) * deg_per_s
    xyz = unit_from_angles(az, np.zeros_like(az))
    s = angular_speed_deg_s(xyz, HZ)
    assert np.isnan(s[:2]).all() and np.isnan(s[-2:]).all()
    assert np.allclose(s[2:-2], deg_per_s, rtol=0.01)


def test_speed_nan_next_to_gap():
    az = np.linspace(0, 5, 100)
    xyz = unit_from_angles(az, np.zeros_like(az))
    xyz[50] = np.nan
    s = angular_speed_deg_s(xyz, HZ)
    assert np.isnan(s[48:53]).all()


def test_detects_synthetic_fixations_and_saccades():
    ev = detect_events(_grid_xyz(), HZ)
    n_fix = len(ev["fixation_durations_s"])
    n_sac = len(ev["saccade_amplitudes_deg"])
    assert n_fix >= 12 and 0.8 <= n_sac / n_fix <= 1.25
    assert 0.24 <= np.median(ev["fixation_durations_s"]) <= FIX_S + 0.01
    assert 0.8 * SACC_DEG <= np.median(ev["saccade_amplitudes_deg"]) <= 1.1 * SACC_DEG


def test_short_fixations_discarded():
    # 30 ms still, then fast motion: no fixation of >= 55 ms exists
    az = np.r_[np.zeros(7), np.linspace(0, 20, 40)]
    ev = detect_events(unit_from_angles(az, np.zeros_like(az)), HZ)
    assert len(ev["fixation_durations_s"]) == 0


def test_artifact_speed_rejected():
    az = np.r_[np.zeros(60), np.full(60, 90.0)]          # 90-degree jump in one sample
    ev = detect_events(unit_from_angles(az, np.zeros_like(az)), HZ)
    assert ev["artifact_mask"].any()
    assert len(ev["saccade_amplitudes_deg"]) == 0


def test_all_nan_input():
    ev = detect_events(np.full((50, 3), np.nan), HZ)
    assert len(ev["fixation_durations_s"]) == 0 and len(ev["saccade_amplitudes_deg"]) == 0
