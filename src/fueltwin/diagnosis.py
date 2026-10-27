"""Fault diagnosis experiments: rule-based baselines and machine learning.

The point of this module is honest validation. With one 60 s recording per
fault setting, windows from the same recording are near copies of each other,
so a shuffled window split mostly measures memory, not diagnosis. Every
experiment here is therefore run under several splits:

* ``random``      shuffled 5-fold over windows (optimistic, shown for contrast)
* ``loro``        leave-one-recording-out: the test setting was never seen
* ``incipient``   train on clear faults (>= 40 % valve travel) and three healthy
                  runs, test on the mild settings and the other healthy runs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import hydraulics as hy

RAW = ["P1", "P2", "P3", "P4", "P5", "Q"]
RAW_DERIVED = RAW + ["dP_filter", "dP_pump", "dP_valve", "dP_nozzle",
                     "P1_std", "P2_std", "P3_std", "P4_std", "P5_std", "Q_std"]
PHYSICS = hy.RESIDUALS
LABELS = ["HC", "F1", "F2", "F3", "F4", "F5"]


def classifiers(seed: int = 0) -> dict:
    return {
        "logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0)),
        "random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "gradient boosting": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed),
    }


# --- splits ------------------------------------------------------------------
def split_random(w: pd.DataFrame, seed: int = 0):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for tr, te in skf.split(w, w["condition"]):
        yield w.index[tr], w.index[te]


def split_loro(w: pd.DataFrame):
    for run in w["run_id"].unique():
        te = w.index[w["run_id"] == run]
        yield w.index.difference(te), te


def split_incipient(w: pd.DataFrame, cut: float = 40.0):
    hc_runs = sorted(w.loc[w["condition"] == "HC", "run_id"].unique())
    train_hc, test_hc = hc_runs[::2], hc_runs[1::2]  # interleave in time: HC-1,3,5 vs HC-2,4,6
    fault = w["condition"] != "HC"
    tr = (fault & (w["severity"] >= cut)) | w["run_id"].isin(train_hc)
    te = (fault & (w["severity"] < cut)) | w["run_id"].isin(test_hc)
    yield w.index[tr], w.index[te]


SPLITS = {"random": split_random, "loro": split_loro, "incipient": split_incipient}


def physics_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Residual features with the section laws fitted on the training windows only."""
    m = hy.ResidualModel.fit(train)
    return m.normalised(train), m.normalised(test)


def run_experiment(w: pd.DataFrame, feature_set: str, model_name: str, split: str,
                   seed: int = 0) -> pd.Series:
    """Out-of-fold predictions for every test window (index-aligned with w)."""
    pred = pd.Series(index=w.index, dtype=object)
    base = classifiers(seed)[model_name]
    for tr, te in SPLITS[split](w):
        if feature_set == "physics":
            xtr, xte = physics_features(w.loc[tr], w.loc[te])
        else:
            cols = {"raw": RAW, "raw+derived": RAW_DERIVED, "P1 only": ["P1"],
                    "clock only": ["clock_h"]}[feature_set]
            xtr, xte = w.loc[tr, cols], w.loc[te, cols]
        clf = clone(base).fit(xtr, w.loc[tr, "condition"])
        pred.loc[te] = clf.predict(xte)
    return pred


def score(y_true: pd.Series, y_pred: pd.Series, runs: pd.Series | None = None) -> dict:
    ok = y_pred.notna()
    y_true, y_pred = y_true[ok], y_pred[ok].astype(str)
    out = {
        "n_windows": int(ok.sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=sorted(y_true.unique())),
    }
    if runs is not None:
        runs = runs[ok]
        vote = pd.DataFrame({"t": y_true, "p": y_pred, "r": runs}).groupby("r").agg(
            t=("t", "first"), p=("p", lambda s: s.value_counts().index[0]))
        out["recordings_correct"] = f"{int((vote.t == vote.p).sum())}/{len(vote)}"
    return out


def signature_direction(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
                        k_detect: float = 1.5, q: float = 0.999) -> pd.Series:
    """Data-driven fault signatures: classify by the *direction* of deviation.

    Each fault's mean z-score vector (relative to the training healthy runs) is
    learned from the training windows. A test window is flagged when its
    deviation is larger than ``k_detect`` times the 99.9th percentile seen on
    healthy training windows, and is assigned to the fault whose direction it
    is closest to (cosine similarity). Because only the direction is used, the
    rule can name faults smaller than any it was trained on.
    """
    hc = train[train["condition"] == "HC"]
    mu, sd = hc[cols].mean(), hc[cols].std()
    ztr, zte = (train[cols] - mu) / sd, (test[cols] - mu) / sd
    thr = k_detect * np.quantile(np.linalg.norm(ztr[train["condition"] == "HC"], axis=1), q)
    dirs = {}
    for f in [c for c in LABELS if c != "HC"]:
        v = ztr[train["condition"] == f].mean()
        dirs[f] = v / np.linalg.norm(v)
    norm = np.linalg.norm(zte, axis=1)
    unit = zte.div(np.maximum(norm, 1e-12), axis=0)
    cos = pd.DataFrame({f: unit @ v for f, v in dirs.items()}, index=test.index)
    return cos.idxmax(axis=1).where(norm > thr, "HC")


def run_direction(w: pd.DataFrame, cols: list[str], split: str) -> pd.Series:
    pred = pd.Series(index=w.index, dtype=object)
    for tr, te in SPLITS[split](w):
        pred.loc[te] = signature_direction(w.loc[tr], w.loc[te], cols).to_numpy()
    return pred


# --- the module's own diagnostic tree (Diagnostic Trees.pdf, page 9) -------------
def tutor_tree(w: pd.DataFrame, base_mean: pd.Series, base_sd: pd.Series, k: float = 4.0) -> pd.Series:
    """Re-implementation of the lecture's diagnostic logic on this data.

    Direction tests ("P3 up/down") use the healthy envelope mean +/- k sd. The
    lecture's absolute thresholds on P3 (1.1, 1.15, 1.2 bar) belong to a
    different rig configuration, so they are applied as written and reported.
    Returns a fault code, "HC", or an ambiguity group such as "F1/F2/F4".
    """
    def direction(col):
        z = (w[col] - base_mean[col]) / base_sd[col]
        return np.where(z > k, 1, np.where(z < -k, -1, 0))

    p3, p4 = direction("P3"), direction("P4")
    out = np.full(len(w), "HC", dtype=object)
    up = p3 == 1
    out[up & (p4 == -1)] = "F3"
    out[up & (p4 == 1)] = "F5"
    out[up & (p4 == 0)] = "F3/F5"
    down = p3 == -1
    p3abs = w["P3"].to_numpy()
    out[down] = "F1/F2/F4"
    out[down & (p3abs < 1.2) & (p3abs > 1.15)] = "F1"
    out[down & (p3abs < 1.15) & (p3abs > 1.1)] = "F2"
    # a fault visible only on other sensors is detected but not isolated by this tree
    others = np.column_stack([direction(c) for c in ["P1", "P2", "P4", "P5", "Q"]])
    out[(p3 == 0) & (others != 0).any(axis=1)] = "detected, not isolated"
    return pd.Series(out, index=w.index)
