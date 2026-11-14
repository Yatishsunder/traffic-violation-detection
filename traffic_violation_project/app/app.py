"""Gradio app. Only LOADS trained models (YOLO best.pt, violation_model.pkl).
It never trains and never runs Optuna."""
import sys
from pathlib import Path

import cv2
import gradio as gr
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import lpr  # noqa: E402
import prediction  # noqa: E402

YOLO_WEIGHTS = ROOT / "models" / "yolo" / "best.pt"
PRED_MODEL = ROOT / "models" / "prediction" / "violation_model.pkl"
for _p in (YOLO_WEIGHTS, PRED_MODEL):
    if not _p.exists():
        raise SystemExit(f"Missing model file: {_p}")

yolo, reader = lpr.load_models(YOLO_WEIGHTS)
bundle = joblib.load(PRED_MODEL)
history = prediction.load_history()

CAVEAT = ("The violation history is synthetic/derived data and the prediction model "
          "performed near chance on held-out plates, so this is a demonstration of the "
          "pipeline, not a reliable forecast.")


def summarize(counts):
    lines = []
    if not counts.get("two_wheeler"):
        lines.append("- No two-wheeler detected.")
    if counts.get("no_helmet"):
        lines.append(f"- **No helmet detected** ({counts['no_helmet']}).")
    if counts.get("mobile_phone"):
        lines.append(f"- Mobile phone detected ({counts['mobile_phone']}). "
                     "This class was unreliable in evaluation; verify manually.")
    if counts.get("earphone"):
        lines.append(f"- Earphone detected ({counts['earphone']}). "
                     "This class was unreliable in evaluation; verify manually.")
    if counts.get("pillion"):
        lines.append(f"- Pillion rider detected ({counts['pillion']}).")
    if not lines:
        lines.append("- No flagged violation classes detected.")
    lines.append("\nThese are model outputs and can be wrong.")
    return "\n".join(lines)


def prediction_text(plates, manual):
    manual = (manual or "").strip()
    if manual:
        candidates = [("manually entered plate", manual)]
    else:
        candidates = [(f"plate {i}", p["text"]) for i, p in enumerate(plates)
                      if p["status"] == "ok"]
    if not candidates:
        if not plates:
            return "No plate detected, so no prediction is possible."
        return ("No plate was read with enough confidence (status 'ok'), so there is "
                "not enough information to look up a violation history.")
    out = []
    for label, text in candidates:
        res = prediction.predict_for_plate(text, history, bundle)
        if res is None:
            out.append(f"**{label} '{text}'**: not found in the violation history, "
                       "so there is not enough information to predict.")
        else:
            out.append(f"**{label} '{text}'**: found in history ({res['n_records']} records, "
                       f"latest {res['last_record_date']}). Model probability of another "
                       f"violation within 30 days: {res['probability']:.2f}.")
    out.append(CAVEAT)
    return "\n\n".join(out)


def analyze(image, manual):
    if image is None:
        return None, "Upload an image first.", [], [], [], ""
    code = cv2.COLOR_RGBA2BGR if image.shape[2] == 4 else cv2.COLOR_RGB2BGR
    bgr = cv2.cvtColor(image, code)

    r = yolo.predict(bgr, conf=lpr.DET_CONF, imgsz=lpr.IMGSZ, verbose=False)[0]
    annotated = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)
    dets = [[r.names[int(c)], round(float(s), 2)] for c, s in zip(r.boxes.cls, r.boxes.conf)]
    counts = {}
    for name, _ in dets:
        counts[name] = counts.get(name, 0) + 1

    plates = lpr.read_plates(bgr, yolo, reader)
    gallery = [(cv2.cvtColor(p["crop"], cv2.COLOR_BGR2RGB),
                f"plate {i}: {p['status']}") for i, p in enumerate(plates) if p["crop"].size]
    rows = []
    for i, p in enumerate(plates):
        text = p["text"] if p["status"] == "ok" else (
            f"{p['text']} (unverified)" if p["text"] else "")
        rows.append([i, p["status"], text, round(p["ocr_conf"], 2), round(p["det_conf"], 2)])

    return annotated, summarize(counts), dets, gallery, rows, prediction_text(plates, manual)


with gr.Blocks(title="Traffic Violation Detection and LPR") as demo:
    gr.Markdown("# Traffic Violation Detection and License Plate Recognition")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Input image")
            manual = gr.Textbox(label="Manual plate (optional; used for the history lookup instead of OCR)")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Detections")
            summary = gr.Markdown()
    det_table = gr.Dataframe(headers=["class", "confidence"], label="Detections")
    gallery = gr.Gallery(label="Plate crops", columns=4)
    plate_table = gr.Dataframe(
        headers=["plate", "status", "OCR text", "OCR conf", "detection conf"],
        label="Plate reading")
    pred_box = gr.Markdown()
    btn.click(analyze, [inp, manual],
              [out_img, summary, det_table, gallery, plate_table, pred_box])

if __name__ == "__main__":
    demo.launch()