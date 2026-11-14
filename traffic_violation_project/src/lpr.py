"""License plate reading: YOLO plate detection -> crop -> EasyOCR.
Importable by the app (load_models, read_plates); also runnable from the terminal.
Never invents text: every result carries a status and only OCR output is returned."""
import argparse
import csv
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models" / "yolo" / "best.pt"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLATE_CLASS = "license_plate"
IMGSZ = 768          # final model was trained/evaluated at 768
DET_CONF = 0.25      # minimum detection confidence
PAD = 0.08           # crop padding as a fraction of box size
OCR_CONF_OK = 0.50   # below this mean OCR confidence -> "uncertain" (starting value, not tuned)
MIN_CHARS = 4        # fewer characters -> "uncertain" (starting value, not tuned)
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def load_models(weights=DEFAULT_WEIGHTS, gpu=True):
    import easyocr
    from ultralytics import YOLO
    return YOLO(str(weights)), easyocr.Reader(["en"], gpu=gpu)


def _preprocess(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    scale = min(4.0, max(1.0, 120.0 / max(h, 1)))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray


def _order(results, height):
    """Left-to-right for one-line plates, top line first for two-line plates."""
    def cy(r):
        return (r[0][0][1] + r[0][2][1]) / 2
    cys = [cy(r) for r in results]
    if len(results) > 1 and (max(cys) - min(cys)) > 0.3 * height:
        mid = (max(cys) + min(cys)) / 2
        return sorted(results, key=lambda r: (0 if cy(r) < mid else 1, r[0][0][0]))
    return sorted(results, key=lambda r: r[0][0][0])


def read_plates(image, model, reader):
    """image: BGR numpy array. Returns a list of dicts, one per detected plate
    (highest detection confidence first); empty list means no plate detected."""
    r = model.predict(image, conf=DET_CONF, imgsz=IMGSZ, verbose=False)[0]
    ih, iw = image.shape[:2]
    out = []
    for box, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls, r.boxes.conf):
        if r.names[int(c)] != PLATE_CLASS:
            continue
        x1, y1, x2, y2 = box
        px, py = (x2 - x1) * PAD, (y2 - y1) * PAD
        x1, y1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        x2, y2 = min(iw, int(x2 + px)), min(ih, int(y2 + py))
        crop = image[y1:y2, x1:x2]
        item = {"box": (x1, y1, x2, y2), "det_conf": float(s), "crop": crop,
                "text": "", "ocr_conf": 0.0, "status": "no_text"}
        if crop.size > 0:
            gray = _preprocess(crop)
            res = reader.readtext(gray, allowlist=ALLOW, detail=1, paragraph=False)
            res = [x for x in res if x[1].strip()]
            if res:
                res = _order(res, gray.shape[0])
                text = "".join(x[1] for x in res).upper()
                text = "".join(ch for ch in text if ch.isalnum())
                conf = sum(float(x[2]) for x in res) / len(res)
                if text:
                    item["text"], item["ocr_conf"] = text, conf
                    ok = conf >= OCR_CONF_OK and len(text) >= MIN_CHARS
                    item["status"] = "ok" if ok else "uncertain"
        out.append(item)
    out.sort(key=lambda d: d["det_conf"], reverse=True)
    return out


def _has_labels(img):
    lbl = img.parent.parent / "labels" / (img.stem + ".txt")
    return lbl.exists() and lbl.read_text().strip() != ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder of images")
    ap.add_argument("--name", required=True, help="output folder name under results\\lpr")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--labeled-only", action="store_true")
    a = ap.parse_args()

    src = Path(a.source) if Path(a.source).is_absolute() else ROOT / a.source
    weights = Path(a.weights) if Path(a.weights).is_absolute() else ROOT / a.weights
    out = ROOT / "results" / "lpr" / a.name
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")
    if out.exists():
        raise SystemExit(f"Output exists, choose another --name: {out}")

    imgs = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if a.labeled_only:
        imgs = [p for p in imgs if _has_labels(p)]
    if not imgs:
        raise SystemExit("No images selected.")

    out.mkdir(parents=True)
    model, reader = load_models(weights)
    rows, counts = [], {}
    for img in imgs:
        im = cv2.imread(str(img))
        if im is None:
            print(f"{img.name}: could not read image")
            continue
        plates = read_plates(im, model, reader)
        if not plates:
            counts["no_plate"] = counts.get("no_plate", 0) + 1
            print(f"{img.name[:40]}: no plate detected")
            rows.append([img.name, "", "no_plate", "", "", ""])
            continue
        for i, p in enumerate(plates):
            cv2.imwrite(str(out / f"{img.stem[:60]}_plate{i}.jpg"), p["crop"])
            counts[p["status"]] = counts.get(p["status"], 0) + 1
            print(f"{img.name[:40]}: plate{i} [{p['status']}] text='{p['text']}' "
                  f"ocr={p['ocr_conf']:.2f} det={p['det_conf']:.2f}")
            rows.append([img.name, i, p["status"], p["text"],
                         f"{p['ocr_conf']:.3f}", f"{p['det_conf']:.3f}"])

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "plate_idx", "status", "text", "ocr_conf", "det_conf"])
        w.writerows(rows)
    print(f"\nImages: {len(imgs)}  Status counts: {counts}")
    print(f"Crops + results.csv saved to: {out}")


if __name__ == "__main__":
    main()