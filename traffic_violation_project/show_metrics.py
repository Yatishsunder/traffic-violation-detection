"""Print the saved classification report and confusion matrix."""

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


MODEL_PATH = "models/prediction/violation_model.pkl"


def main():
    bundle = joblib.load(MODEL_PATH)
    saved_metrics = bundle["test_metrics"]
    saved_matrix = np.asarray(saved_metrics["confusion_matrix"], dtype=int)

    # Recreate the held-out labels represented by the saved confusion matrix.
    actual = np.repeat([0, 1], saved_matrix.sum(axis=1))
    predicted = np.concatenate(
        [
            np.repeat([0, 1], saved_matrix[0]),
            np.repeat([0, 1], saved_matrix[1]),
        ]
    )

    print("Classification report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["No repeat violation (0)", "Repeat violation (1)"],
            digits=2,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))


if __name__ == "__main__":
    main()
