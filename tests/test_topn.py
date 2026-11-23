import numpy as np
import pandas as pd
import pytest

from topn import MODELS_TOPN, N_CANDIDATES, TopNByAUC, selected_features


def _table(n_people=20, n_noise=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n_people)
    X = pd.DataFrame(rng.normal(size=(len(y), n_noise)), columns=[f"noise{i}" for i in range(n_noise)])
    X.insert(3, "signal", y * 2.0 + rng.normal(scale=0.3, size=len(y)))
    return X, y, np.repeat(np.arange(n_people), 2)


def test_selector_keeps_the_feature_that_separates_the_classes():
    X, y, _ = _table()
    sel = TopNByAUC(n=1).fit(X, y)
    assert list(X.columns[sel.get_support()]) == ["signal"]
    assert sel.transform(X).shape == (len(X), 1)


def test_selector_scores_both_directions_and_caps_n():
    X, y, _ = _table()
    X["signal"] = -X["signal"]                     # lower in high load: still informative
    assert list(X.columns[TopNByAUC(n=1).fit(X, y).get_support()]) == ["signal"]
    assert TopNByAUC(n=500).fit(X, y).get_support().sum() == X.shape[1]


def test_selector_ignores_blanks_when_scoring():
    X, y, _ = _table()
    X.loc[::5, "signal"] = np.nan
    assert np.flatnonzero(TopNByAUC(n=1).fit(X.to_numpy(), y).get_support()).tolist() == [3]


def test_candidates_and_spaces():
    assert N_CANDIDATES == (5, 10, 20, 40, 80, 140)
    for spec in MODELS_TOPN.values():
        assert list(spec["space"]["select__n"]) == list(N_CANDIDATES)


@pytest.mark.parametrize("name", ["Logistic Regression", "XGBoost"])
def test_pipelines_fit_with_a_chosen_n(name):
    X, y, _ = _table()
    model = MODELS_TOPN[name]["factory"]().set_params(select__n=2).fit(X, y)
    assert model.predict_proba(X).shape == (len(X), 2)
    assert "signal" in selected_features(model, list(X.columns))
    assert len(selected_features(model, list(X.columns))) == 2
