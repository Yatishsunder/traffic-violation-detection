"""Train the violation-prediction model on prediction_data\\historical.csv.
Leakage-safe: split by plate, last record per plate excluded, features use only
information known at the time of a record. The data is synthetic/derived.
The Gradio app only LOADS the saved model (and may import load_history / predict_for_plate)."""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "prediction_data" / "historical.csv"
MODEL_PATH = ROOT / "models" / "prediction" / "violation_model.pkl"
METRICS_PATH = ROOT / "results" / "prediction_metrics.json"
SEED = 42
TARGET = "future_violation_within_30_days"
CAT = ["violation_type", "location"]
NUM = ["previous_violation_count", "days_since_previous_violation",
       "month", "day_of_week", "hour"]
FEATURES = CAT + NUM


def normalize_plate(text):
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def load_history():
    df = pd.read_csv(CSV)
    df["plate_key"] = df["plate_number"].map(normalize_plate)
    df = df.sort_values(["plate_key", "date", "time", "record_id"]).reset_index(drop=True)
    d = pd.to_datetime(df["date"], format="%Y-%m-%d")
    t = pd.to_datetime(df["time"], format="%H:%M:%S")
    df["month"] = d.dt.month
    df["day_of_week"] = d.dt.dayofweek
    df["hour"] = t.dt.hour
    return df


def build(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def predict_for_plate(plate_text, history, bundle):
    """Returns None if the plate is not in the history (no guess is made).
    Otherwise uses that plate's latest record as the model input."""
    key = normalize_plate(plate_text)
    rows = history[history["plate_key"] == key]
    if rows.empty:
        return None
    last = rows.iloc[[-1]]
    prob = float(bundle["pipeline"].predict_proba(last[bundle["features"]])[0, 1])
    return {"plate": str(last["plate_number"].iloc[0]), "n_records": int(len(rows)),
            "last_record_date": str(last["date"].iloc[0]), "probability": prob}


def test_metrics(pipe, X, y):
    pred = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main():
    if not CSV.exists():
        raise SystemExit(f"Missing: {CSV}")
    for p in (MODEL_PATH, METRICS_PATH):
        if p.exists():
            raise SystemExit(f"Already exists, not overwriting: {p}")

    df = load_history()
    is_last = df.groupby("plate_key").cumcount(ascending=False) == 0
    print(f"Rows: {len(df)}  plates: {df['plate_key'].nunique()}")
    print(f"Last record per plate excluded: {int(is_last.sum())} rows "
          f"(positives among them: {int(df.loc[is_last, TARGET].sum())})")
    data = df[~is_last].reset_index(drop=True)
    X, y, g = data[FEATURES], data[TARGET], data["plate_key"]

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, g))
    Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], g.iloc[tr]
    Xte, yte, gte = X.iloc[te], y.iloc[te], g.iloc[te]
    assert not (set(gtr) & set(gte)), "plate overlap between train and test"
    print(f"Train: {len(Xtr)} rows / {gtr.nunique()} plates, positive rate {ytr.mean():.3f}")
    print(f"Test : {len(Xte)} rows / {gte.nunique()} plates, positive rate {yte.mean():.3f}")

    models = {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                random_state=SEED),
    }
    cv_summary = {}
    print("\n=== 5-fold CV grouped by plate (training plates only), mean +- std ===")
    for name, m in models.items():
        r = cross_validate(build(m), Xtr, ytr, groups=gtr, cv=GroupKFold(n_splits=5),
                           scoring={"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"})
        cv_summary[name] = {k: {"mean": float(np.mean(r[f"test_{k}"])),
                                "std": float(np.std(r[f"test_{k}"]))}
                            for k in ("accuracy", "f1", "roc_auc")}
        s = cv_summary[name]
        print(f"{name:22s} acc {s['accuracy']['mean']:.3f}+-{s['accuracy']['std']:.3f}  "
              f"f1 {s['f1']['mean']:.3f}+-{s['f1']['std']:.3f}  "
              f"auc {s['roc_auc']['mean']:.3f}+-{s['roc_auc']['std']:.3f}")

    best = max(("logistic_regression", "random_forest"),
               key=lambda n: cv_summary[n]["roc_auc"]["mean"])
    print(f"\nSelected by CV ROC AUC (rule fixed beforehand): {best}")

    final = build(models[best]).fit(Xtr, ytr)
    baseline = build(DummyClassifier(strategy="most_frequent")).fit(Xtr, ytr)
    res = {best: test_metrics(final, Xte, yte),
           "majority_baseline": test_metrics(baseline, Xte, yte)}
    print("\n=== Held-out test plates (used once, after selection) ===")
    for name, m in res.items():
        print(f"{name:22s} acc {m['accuracy']:.3f}  f1 {m['f1']:.3f}  "
              f"auc {m['roc_auc']:.3f}  confusion {m['confusion_matrix']}")

    bundle = {"pipeline": final, "features": FEATURES, "target": TARGET, "model_name": best,
              "note": "Trained on synthetic/derived historical.csv; plate-based split; "
                      "last record per plate excluded.",
              "test_metrics": res[best]}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "seed": SEED, "n_rows_used": int(len(data)), "n_plates": int(g.nunique()),
        "train_plates": int(gtr.nunique()), "test_plates": int(gte.nunique()),
        "cv_grouped_by_plate": cv_summary, "selected_model": best,
        "held_out_test": res}, indent=2))
    print(f"\nSaved model:   {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()