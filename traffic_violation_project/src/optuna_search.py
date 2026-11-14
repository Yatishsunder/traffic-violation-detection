import argparse
import gc
import json
from pathlib import Path

import optuna
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset" / "processed" / "data.yaml"
PRETRAINED = ROOT / "models" / "yolo" / "pretrained" / "yolo11n.pt"
RUNS = ROOT / "models" / "yolo" / "runs" / "optuna"
OUT = ROOT / "optuna"


def make_objective(study_name, epochs):
    def objective(trial):
        p = {
            "lr0": trial.suggest_float("lr0", 1e-4, 1e-2, log=True),
            "lrf": trial.suggest_float("lrf", 0.01, 0.5, log=True),
            "momentum": trial.suggest_float("momentum", 0.8, 0.98),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "imgsz": trial.suggest_categorical("imgsz", [512, 640, 768]),
            "optimizer": trial.suggest_categorical("optimizer", ["SGD", "AdamW"]),
            "mosaic": trial.suggest_float("mosaic", 0.0, 1.0),
            "scale": trial.suggest_float("scale", 0.2, 0.9),
        }
        run_name = f"{study_name}_trial_{trial.number}"
        run_dir = RUNS / run_name
        if run_dir.exists():
            raise SystemExit(f"Run folder exists, refusing to reuse: {run_dir}")

        model = None
        try:
            model = YOLO(str(PRETRAINED))
            model.train(
                data=str(DATA), epochs=epochs, batch=16, project=str(RUNS),
                name=run_name, exist_ok=False, seed=0, workers=2,
                plots=False, verbose=False, **p,
            )
            best = run_dir / "weights" / "best.pt"
            if not best.exists():
                raise optuna.TrialPruned("best.pt missing")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            m = YOLO(str(best)).val(
                data=str(DATA), split="val", imgsz=p["imgsz"],
                plots=False, verbose=False,
            )
            trial.set_user_attr("map50", float(m.box.map50))
            trial.set_user_attr("run_dir", str(run_dir))
            return float(m.box.map)
        finally:
            model = None
            gc.collect()
            torch.cuda.empty_cache()

    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--trials", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    db = OUT / f"{a.study}.db"
    if db.exists():
        raise SystemExit(f"Study already exists, choose another --study: {db}")
    if not DATA.exists() or not PRETRAINED.exists():
        raise SystemExit("data.yaml or pretrained weights not found.")

    study = optuna.create_study(
        study_name=a.study,
        direction="maximize",
        storage=f"sqlite:///{db.as_posix()}",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    # trial 0 = Ultralytics default values at the same epoch budget (reference)
    study.enqueue_trial({
        "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
        "imgsz": 640, "optimizer": "SGD", "mosaic": 1.0, "scale": 0.5,
    })

    log_path = OUT / f"{a.study}_trial_log.jsonl"

    def log_cb(study_, t):
        rec = {"number": t.number, "state": str(t.state), "value": t.value,
               "params": t.params, "user_attrs": t.user_attrs}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[trial {t.number}] {t.state} mAP50-95={t.value} params={t.params}")

    study.optimize(
        make_objective(a.study, a.epochs), n_trials=a.trials,
        callbacks=[log_cb], catch=(torch.cuda.OutOfMemoryError,),
    )

    done = [t for t in study.trials if t.value is not None]
    if not done:
        raise SystemExit("No trial completed.")
    best = max(done, key=lambda t: t.value)
    ref = next((t for t in done if t.number == 0), None)

    summary = {
        "study": a.study, "epochs_per_trial": a.epochs,
        "objective": "val mAP50-95 (val used for selection, so optimistic)",
        "best_trial": best.number, "best_value": best.value,
        "best_params": best.params,
        "reference_trial0_value": ref.value if ref else None,
        "n_completed": len(done), "n_total": len(study.trials),
    }
    (OUT / f"{a.study}_best.json").write_text(json.dumps(summary, indent=2))
    study.trials_dataframe().to_csv(OUT / f"{a.study}_trials.csv", index=False)

    print("\n=== Ranking (val mAP50-95) ===")
    for t in sorted(done, key=lambda t: t.value, reverse=True):
        print(f"trial {t.number}: {t.value:.4f}  {t.params}")
    print(f"\nBest: trial {best.number}  reference (trial 0): "
          f"{ref.value if ref else 'n/a'}")
    print(f"Saved to: {OUT}")


if __name__ == "__main__":
    main()