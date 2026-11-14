
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

bundle = joblib.load("models/prediction/violation_model.pkl")
saved = bundle["test_metrics"]
cm = np.array(saved["confusion_matrix"])

# Rebuild the actual and predicted labels from the saved confusion matrix.
y_true = np.repeat([0, 1], cm.sum(axis=1))
y_pred = np.concatenate([
    np.repeat([0, 1], cm[0]),
    np.repeat([0, 1], cm[1]),
])

print("Classification report:")
print(classification_report(
    y_true, y_pred,
    labels=[0, 1],
    target_names=["No repeat violation (0)", "Repeat violation (1)"],
    digits=2,
))

print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_true, y_pred, labels=[0, 1]))