import numpy as np
import pandas as pd

from checks import (
    manipulation_check, missingness_report, redundancy_check, sanity_check, sanity_passed,
)


def _rows(fix_ms, n_fix, n_sac):
    """Two participants with A1 and A4 as given; A2/A3 are filled with healthy values."""
    df = pd.DataFrame({
        "participant": [1, 1, 2, 2], "activity": [1, 4, 1, 4],
        "fixation_median_ms": fix_ms, "n_fixations": n_fix, "n_saccades": n_sac,
    })
    mid = pd.DataFrame({"participant": [1, 1], "activity": [2, 3], "fixation_median_ms": [250, 250],
                        "n_fixations": [100, 100], "n_saccades": [100, 100]})
    return pd.concat([df, mid], ignore_index=True)


def test_sanity_pass():
    s = sanity_check(_rows([250, 260, 240, 270], [100, 100, 100, 100], [100, 95, 105, 100]))
    assert sanity_passed(s) and s.passed.all()


def test_sanity_fails_if_one_activity_fails():
    # A4 has far more saccades than fixations (noise) -> fails, even though pooled ratio is ~1.1
    s = sanity_check(_rows([250, 260, 240, 270], [100, 60, 100, 60], [80, 100, 80, 100]))
    assert not sanity_passed(s)
    assert s.set_index("activity").loc[4, "ratio_ok"] == False  # noqa: E712


def test_sanity_fails_if_an_activity_has_no_rows():
    rows = _rows([250, 260, 240, 270], [100, 100, 100, 100], [100, 95, 105, 100])
    s = sanity_check(rows[rows.activity != 3])
    assert not sanity_passed(s)
    row = s.set_index("activity").loc[3]
    assert row["recordings"] == 0 and row["passed"] == False  # noqa: E712


def test_manipulation_check():
    ann = pd.DataFrame({"participant": np.repeat([1, 2, 3], 4), "activity": [1, 2, 3, 4] * 3,
                        "mean": [10, 20, 30, 40, 15, 25, 35, 45, 12, 18, 33, 50]})
    out = manipulation_check(ann, [1, 2, 3])
    assert out["share_a4_above_a1"] == 1.0 and out["n"] == 3


def test_redundancy_check():
    rows = pd.DataFrame({"saccade_peak_velocity": [1, 2, 3, 4.0], "saccade_mean_velocity": [2, 4, 6, 8.0],
                         "saccade_amplitude": [1, 2, 3, 4.0], "saccade_duration_ms": [4, 3, 2, 1.0]})
    out = redundancy_check(rows).set_index("pair")
    assert out.loc["saccade_peak_velocity~saccade_mean_velocity", "spearman_rho"] == 1.0
    assert out.loc["saccade_amplitude~saccade_duration_ms", "spearman_rho"] == -1.0


def test_missingness_report():
    rows = pd.DataFrame({"activity": [1, 1, 4, 4], "pupil_mean": [1.0, np.nan, np.nan, np.nan]})
    out = missingness_report(rows, ["pupil_mean"]).set_index(["activity", "feature"])
    assert out.loc[(1, "pupil_mean"), "n_nan"] == 1 and out.loc[(1, "pupil_mean"), "share_nan"] == 0.5
    assert out.loc[(4, "pupil_mean"), "share_nan"] == 1.0
    assert out.loc[(2, "pupil_mean"), "recordings"] == 0
