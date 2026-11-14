"""Train YOLO11n on the processed IMVT dataset.
Run explicitly from the terminal. The Gradio app never imports or runs this file."""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed" / "data.yaml"
PRETRAINED = ROOT / "models" / "yolo" / "pretrained" / "yolo11n.pt"
RUNS = ROOT / "models" / "yolo" / "runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name, e.g. smoke_test")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    for p in (DATA, PRETRAINED):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")
    if (RUNS / args.name).exists():
        raise SystemExit(f"Run folder already exists: {RUNS / args.name}. Choose a different --name.")

    model = YOLO(str(PRETRAINED))
    model.train(
        data=str(DATA),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(RUNS),
        name=args.name,
        exist_ok=False,
        seed=0,
        deterministic=True,
        workers=2,
        plots=True,
    )
    print("\nDone. Weights are in:", RUNS / args.name / "weights")


if __name__ == "__main__":
    main()