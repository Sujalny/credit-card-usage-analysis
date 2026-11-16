"""Predict monthly spending for customer profiles using a saved model."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import CATEGORICAL, NUMERIC, ROOT, encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Profile CSV: customer_id,age,annual_income,gender,city,card_type")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "predictions.csv")
    args = parser.parse_args()
    import tensorflow as tf
    frame = pd.read_csv(args.input)
    required = ["customer_id", *NUMERIC, *CATEGORICAL]
    if not set(required).issubset(frame.columns) or frame.empty:
        parser.error(f"A nonempty CSV with these columns is required: {', '.join(required)}")
    for column in CATEGORICAL:
        frame[column] = frame[column].astype("string").str.strip().str.title().replace("", pd.NA).fillna("Unknown")
    for column in NUMERIC:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not (np.isfinite(frame[NUMERIC]).all(axis=1) & frame.age.between(18, 100) & frame.annual_income.gt(0) & frame.customer_id.notna()).all():
        parser.error("Profiles need a customer_id, age 18–100, and positive finite annual_income.")
    metadata = json.loads((args.model_dir / "preprocessing.json").read_text(encoding="utf-8"))
    features, _ = encode(frame, metadata)
    model = tf.keras.models.load_model(args.model_dir / "spending_model.keras", compile=False)
    frame["predicted_monthly_spend_inr"] = np.maximum(model(features, training=False).numpy().ravel() * metadata["target_scale"], 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Saved {len(frame)} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
