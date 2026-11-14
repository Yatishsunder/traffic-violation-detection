import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def has_labels(img):
    lbl = img.parent.parent / "labels" / (img.stem + ".txt")
    return lbl.exists() and lbl.read_text().strip() != ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="folder of images")
    ap.add_argument("--name", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--labeled-only", action="store_true",
                    help="only images that have a non-empty label file")
    a = ap.parse_args()

    weights = resolve(a.weights)
    src = resolve(a.source)
    out = ROOT / "results" / "predictions" / a.name

    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")
    if out.exists():
        raise SystemExit(f"Output exists, choose another --name: {out}")

    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if a.labeled_only:
        imgs = [p for p in imgs if has_labels(p)]
    if not imgs:
        raise SystemExit("No images selected.")

    out.mkdir(parents=True)
    model = YOLO(str(weights))
    totals = {}

    for img in imgs:
        r = model.predict(str(img), conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        ok = cv2.imwrite(str(out / img.name), r.plot())
        dets = [(r.names[int(c)], float(s)) for c, s in zip(r.boxes.cls, r.boxes.conf)]
        for n, _ in dets:
            totals[n] = totals.get(n, 0) + 1
        txt = ", ".join(f"{n} {s:.2f}" for n, s in dets) or "no detections"
        print(f"{img.name}{'' if ok else ' (SAVE FAILED)'}: {txt}")

    print(f"\nImages processed: {len(imgs)}  conf>={a.conf}")
    print("Detection counts:", totals)
    print(f"Annotated images saved to: {out}")


if __name__ == "__main__":
    main()
    