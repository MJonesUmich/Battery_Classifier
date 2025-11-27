"""Run inference on summarized battery features using the pre-trained model.

This script wires together two artifacts produced earlier:

- `artifacts/logreg_sample.csv`: feature rows created by `prepare_logreg_dataset`.
- `notebooks/artifacts/logistic_model.pkl`: the sklearn LogisticRegression model.

It loads both, ensures the feature matrix matches what the model expects, then
emits the predicted chemistry label plus per-class probabilities.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DEFAULT_CLASSES: List[str] = ["LCO", "LFP", "NCA", "NMC"]
DEFAULT_DATA_PATH = Path("artifacts/logreg_sample.csv")
DEFAULT_MODEL_PATH = Path("notebooks/artifacts/logistic_model.pkl")
LABEL_COLUMN = "chemistry"


def _load_model(model_path: Path):
    with model_path.open("rb") as f:
        return pickle.load(f)


def _build_class_mapping(model_classes: Iterable[int]) -> Dict[int, str]:
    model_classes = list(model_classes)
    if len(model_classes) == len(DEFAULT_CLASSES):
        return dict(zip(model_classes, DEFAULT_CLASSES))
    # Fallback: just echo the integer class ids as strings.
    return {cls: str(cls) for cls in model_classes}


def predict(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    label_column: str = LABEL_COLUMN,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    y_true = df[label_column] if label_column in df.columns else None
    X = df.drop(columns=[label_column], errors="ignore")

    model = _load_model(model_path)
    if getattr(model, "n_features_in_", None) != X.shape[1]:
        raise ValueError(
            f"Feature count mismatch: CSV has {X.shape[1]}, "
            f"model expects {getattr(model, 'n_features_in_', 'unknown')}."
        )

    X_array = X.to_numpy()
    predictions = model.predict(X_array)
    probabilities = model.predict_proba(X_array)

    class_mapping = _build_class_mapping(model.classes_)
    pred_labels = [class_mapping.get(cls, str(cls)) for cls in predictions]
    prob_columns = [f"prob_{class_mapping.get(cls, str(cls))}" for cls in model.classes_]
    prob_df = pd.DataFrame(probabilities, columns=prob_columns)

    result = pd.DataFrame(
        {
            "prediction_raw": predictions,
            "prediction_label": pred_labels,
        }
    )
    if y_true is not None:
        result["ground_truth"] = y_true

    return pd.concat([df, result, prob_df], axis=1)


def main(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    output_path: str | Path = "artifacts/logreg_sample_predictions.csv",
) -> pd.DataFrame:
    """Convenience wrapper so users can call `main('path/to.csv')` directly."""

    data_path = Path(data_path)
    model_path = Path(model_path)
    output_path = Path(output_path)

    predictions = predict(data_path=data_path, model_path=model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    print(
        predictions[
            ["prediction_label", "prediction_raw"]
            + [col for col in predictions.columns if col.startswith("prob_")]
        ]
    )

    return predictions


if __name__ == "__main__":
    main()

