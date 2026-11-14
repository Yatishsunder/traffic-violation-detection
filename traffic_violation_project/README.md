# AI-Based Traffic Violation Detection and License Plate Recognition with Traffic Violation Prediction

A final-semester ML project with three parts:

1. **Detection:** YOLO11n detects two-wheelers, riders, helmets/no-helmet, pillion riders, mobile phones, earphones and license plates.
2. **License plate recognition:** the detected plate is cropped and read with EasyOCR.
3. **Violation prediction:** a small scikit-learn model estimates the chance of another violation within 30 days for a plate found in a historical record file.

A Gradio app combines all three. The app only loads trained models; it never trains or runs Optuna.

## Setup (Windows, Python 3.12)

Install PyTorch first (CUDA 12.4 build used in this project), then the rest:

```
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

All results below were produced on an RTX 4050 Laptop GPU (6 GB).

## Project layout

```
dataset/raw/IMVTDataset      original dataset (untouched)
dataset/processed            cleaned copy used for training (data.yaml inside)
models/yolo/best.pt          final detection model
models/yolo/runs/            training runs (baseline_640, final_768, optuna trials)
models/prediction/           violation_model.pkl
prediction_data/historical.csv
src/                         train.py, train_final.py, optuna_search.py, evaluate.py,
                             predict.py, lpr.py, prediction.py
app/app.py                   Gradio app
results/  optuna/            metrics, predictions, study files
```

[Add the name of the script used to build dataset/processed from dataset/raw here.]

## Dataset (IMVT)

- Original: 10 classes; train 5785 / val 346 / test 206 images, but only 186 / 44 / 21 images actually have labels (the rest are empty background).
- Processing: polygon labels converted to bounding boxes; classes `b` and `object` dropped (1 instance each, train only); 2 tiny boxes dropped; 60 empty images added to train.
- Processed splits: train 246 images (1336 instances), val 44 images (340 instances), test 21 images (168 instances), 8 classes.
- No leakage between splits among labeled images (checked by source name and MD5).
- The visual check of the polygon-to-box conversion previews was not confirmed.

## Detection results

Produced with `src/evaluate.py` (same script for both models; final model evaluated at imgsz 768).

| Model | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---|---|---|---|
| baseline_640 (YOLO11n, 100 epochs, defaults) | 0.533 | 0.286 | 0.547 | 0.274 |
| final_768 (Optuna best params, 100 epochs) | 0.551 | 0.293 | 0.629 | 0.363 |

Notes:
- The val gain is small (44 images) and within the noise seen in this project. The test gain is larger, but the test set has only 21 images.
- The tuned run changed several settings at once, so the gain cannot be attributed to a single one.
- `license_plate` is the strongest class that matters downstream (test mAP50 0.866, val 0.760). `no_helmet` precision improved over the baseline (val 0.389 to 0.507, test 0.433 to 0.586). `helmet` recall dropped (val 0.716 to 0.600, test 0.773 to 0.545).
- `mobile_phone` and `earphone` are effectively unreliable (very few training examples).
- The checkpoint was chosen using the same 44 validation images, so validation numbers are somewhat optimistic.

## Hyperparameter search (Optuna)

7 trials of 30 epochs each, objective = validation mAP50-95, with trial 0 fixed to the default values as a reference.

- Trial 0 (defaults): 0.254. Best trial (3): 0.277, with imgsz 768 and AdamW. The three best trials all used imgsz 768 and lie within 0.009 of each other.
- 30-epoch scores are not comparable with the 100-epoch baseline.
- The final model was then trained for 100 epochs with the best trial's parameters (`src/train_final.py`, reading `optuna/v1_best.json`).
- With only 7 trials the search is close to random sampling; the improvement should be read as small.

## License plate recognition

On the 21 labeled test images, the model detected 31 plates (3 images had no plate detected). EasyOCR statuses: 7 `ok`, 17 `uncertain`, 4 `no_text`.

- `ok` means OCR confidence of at least 0.50 and at least 4 characters. It does not mean the text is correct.
- No ground-truth plate text was collected, so no OCR accuracy is reported. Most plates came back with low OCR confidence, and some `ok` results look implausible.
- The app only uses a plate for the history lookup when its status is `ok`, and shows other readings as unverified. OCR thresholds are starting values and were not tuned.

## Violation prediction

Data: `prediction_data/historical.csv` is **synthetic/derived** data (640 rows, 100 plates). The target is whether another violation occurs within 30 days.

- Features: violation type, location, month, day of week, hour, previous violation count, days since previous violation. The last record of each plate is excluded (its label is always 0 only because the data ends there), leaving 540 rows.
- Split by plate: 80 training plates, 20 held-out plates. Model selection by 5-fold CV grouped by plate; the held-out plates were used once.

| | CV accuracy | CV ROC AUC | held-out accuracy | held-out ROC AUC |
|---|---|---|---|---|
| majority baseline | 0.600 | 0.500 | 0.580 | 0.500 |
| logistic regression (selected) | 0.630 | 0.648 | 0.530 | 0.511 |
| random forest | 0.598 | 0.638 | not evaluated | not evaluated |

The model performs near chance on held-out plates and below the majority baseline in accuracy. The app's probability is an illustration of the pipeline, not a reliable forecast.

## Running the project

Run from the project root:

```
python src\train.py --name baseline_640 --epochs 100 --imgsz 640 --batch 16
python src\optuna_search.py --study v1 --trials 7 --epochs 30
python src\train_final.py --name final_768
python src\evaluate.py --weights models\yolo\best.pt --split test --name final_768_check --imgsz 768
python src\lpr.py --source dataset\processed\test\images --name test_labeled --labeled-only
python src\prediction.py
python app\app.py
```

The training scripts refuse to reuse an existing run or output name, so choose a new name for each run. The app opens a local address (usually http://127.0.0.1:7860).

## Limitations

- Very small labeled validation (44) and test (21) sets; results have wide uncertainty.
- Weak classes (`mobile_phone`, `earphone`) are not reliable.
- Plate OCR is weak and was not quantitatively evaluated.
- The prediction model uses synthetic data and is near chance.
- Not tested on photos from outside the IMVT dataset.
### Exploratory: next-location prediction

`src/location_prediction.py` tried to predict the location of a plate's next recorded violation (8 locations) with the same plate-based split and CV selection. Held-out test (20 plates, 100 rows): top-1 accuracy 0.120 and top-3 accuracy 0.290 for the selected logistic regression, against 0.130 / 0.350 for the majority baseline and 0.110 top-1 for "same place as the last record" (uniform chance: 0.125 / 0.375). The model shows no skill, so it is not used in the app.