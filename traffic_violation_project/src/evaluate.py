"""Evaluate saved YOLO weights on the processed val or test split. No training happens here."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "processed" / "data.yaml"
RESULTS = ROOT / "results"


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--name", required=True, help="label for this evaluation, e.g. baseline_640")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    RESULTS.mkdir(exist_ok=True)
    out_json = RESULTS / f"eval_{args.name}_{args.split}.json"
    if out_json.exists():
        raise SystemExit(f"Already exists: {out_json}. Choose a different --name.")

    model = YOLO(str(weights))
    m = model.val(data=str(DATA), split=args.split, imgsz=args.imgsz, device=0, plots=True,
                  project=str(RESULTS / "yolo_eval"), name=f"{args.name}_{args.split}",
                  verbose=False)

    p, r = float(m.box.mp), float(m.box.mr)
    overall = {"precision": p, "recall": r, "f1": f1(p, r),
               "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)}
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        cp, cr, ap50, ap = m.box.class_result(i)
        per_class[m.names[int(c)]] = {"precision": float(cp), "recall": float(cr),
                                      "f1": f1(float(cp), float(cr)),
                                      "mAP50": float(ap50), "mAP50-95": float(ap)}

    out = {"weights": str(weights), "split": args.split, "imgsz": args.imgsz,
           "overall": overall, "per_class": per_class}
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== {args.name} | split: {args.split} ===")
    print("overall:", {k: round(v, 3) for k, v in overall.items()})
    for name, v in per_class.items():
        print(f"{name:14s}", {k: round(x, 3) for k, x in v.items()})
    print("\nSaved:", out_json)


if __name__ == "__main__":
    main()