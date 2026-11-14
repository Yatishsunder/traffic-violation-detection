"""Exploratory: predict the LOCATION of a plate's next recorded violation.
Synthetic/derived data. Plate-based split; the last record per plate has no next
record and is dropped. Features use only information up to the current record."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import prediction

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "prediction" / "location_model.pkl"
METRICS_PATH = ROOT / "results" / "location_metrics.json"
SEED = 42


def build(model, cat, num):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(pipe, X, y):
    proba = pipe.predict_proba(X)
    classes = np.asarray(pipe.classes_)
    top1 = classes[np.argmax(proba, axis=1)]
    top3 = classes[np.argsort(-proba, axis=1)[:, :3]]
    yv = y.to_numpy()
    return {"top1": float(np.mean(top1 == yv)),
            "top3": float(np.mean([a in row for a, row in zip(yv, top3)]))}


def main():
    for p in (MODEL_PATH, METRICS_PATH):
        if p.exists():
            raise SystemExit(f"Already exists, not overwriting: {p}")

    df = prediction.load_history()
    locs = sorted(df["location"].unique())
    count_cols = []
    for loc in locs:
        col = "n_" + loc.replace(" ", "_")
        df[col] = (df["location"] == loc).groupby(df["plate_key"]).cumsum()
        count_cols.append(col)
    df["next_location"] = df.groupby("plate_key")["location"].shift(-1)
    data = df.dropna(subset=["next_location"]).reset_index(drop=True)

    cat = ["violation_type", "location"]
    num = (["previous_violation_count", "days_since_previous_violation",
            "month", "day_of_week", "hour"] + count_cols)
    feats = cat + num
    X, y, g = data[feats], data["next_location"], data["plate_key"]
    print(f"Rows with a next record: {len(data)}  plates: {g.nunique()}  "
          f"locations: {len(locs)} (uniform chance = {1 / len(locs):.3f})")

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                    random_state=SEED).split(X, y, g))
    Xtr, ytr, gtr = X.iloc[tr], y.iloc[tr], g.iloc[tr]
    Xte, yte, gte = X.iloc[te], y.iloc[te], g.iloc[te]
    assert not (set(gtr) & set(gte)), "plate overlap between train and test"
    print(f"Train: {len(Xtr)} rows / {gtr.nunique()} plates   "
          f"Test: {len(Xte)} rows / {gte.nunique()} plates")

    models = {
        "majority_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000),
        "random_forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                random_state=SEED),
    }
    cv = {}
    print("\n=== 5-fold CV grouped by plate (training plates only), top-1 accuracy ===")
    for name, m in models.items():
        r = cross_validate(build(m, cat, num), Xtr, ytr, groups=gtr,
                           cv=GroupKFold(n_splits=5), scoring="accuracy")
        cv[name] = {"mean": float(np.mean(r["test_score"])),
                    "std": float(np.std(r["test_score"]))}
        print(f"{name:22s} {cv[name]['mean']:.3f} +- {cv[name]['std']:.3f}")

    best = max(("logistic_regression", "random_forest"), key=lambda n: cv[n]["mean"])
    print(f"\nSelected by CV top-1 (rule fixed beforehand): {best}")

    final = build(models[best], cat, num).fit(Xtr, ytr)
    base = build(models["majority_baseline"], cat, num).fit(Xtr, ytr)
    res = {best: evaluate(final, Xte, yte),
           "majority_baseline": evaluate(base, Xte, yte),
           "same_as_last_location": {"top1": float(np.mean(Xte["location"].to_numpy()
                                                          == yte.to_numpy())),
                                     "top3": None}}
    print("\n=== Held-out test plates (used once, after selection) ===")
    for name, m in res.items():
        t3 = "n/a" if m["top3"] is None else f"{m['top3']:.3f}"
        print(f"{name:24s} top-1 {m['top1']:.3f}   top-3 {t3}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "features": feats, "model_name": best,
                 "note": "Exploratory; synthetic/derived data; plate-based split.",
                 "test_metrics": res[best]}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "seed": SEED, "n_rows": int(len(data)), "n_locations": len(locs),
        "train_plates": int(gtr.nunique()), "test_plates": int(gte.nunique()),
        "cv_top1_grouped_by_plate": cv, "selected_model": best,
        "held_out_test": res}, indent=2))
    print(f"\nSaved model:   {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
    