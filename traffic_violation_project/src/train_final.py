"""Final YOLO11n training using the best Optuna params (read from optuna\\<study>_best.json).
Run explicitly from the terminal. The Gradio app never imports or runs this file."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed" / "data.yaml"
PRETRAINED = ROOT / "models" / "yolo" / "pretrained" / "yolo11n.pt"
RUNS = ROOT / "models" / "yolo" / "runs"
OPTUNA = ROOT / "optuna"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, e.g. final_768")
    ap.add_argument("--study", default="v1", help="Optuna study whose best params to use")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    best_json = OPTUNA / f"{args.study}_best.json"
    for p in (DATA, PRETRAINED, best_json):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")
    if (RUNS / args.name).exists():
        raise SystemExit(f"Run folder already exists: {RUNS / args.name}. Choose a different --name.")

    summary = json.loads(best_json.read_text())
    params = summary["best_params"]
    print(f"Using params from {best_json.name} (trial {summary['best_trial']}):")
    for k, v in params.items():
        print(f"  {k}: {v}")

    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        batch=args.batch,
        device=0,
        project=str(RUNS),
        name=args.name,
        exist_ok=False,
        seed=0,
        deterministic=True,
        workers=2,
        plots=True,
        **params,
    )
    print("\nDone. Weights are in:", RUNS / args.name / "weights")


if __name__ == "__main__":
    main()
    