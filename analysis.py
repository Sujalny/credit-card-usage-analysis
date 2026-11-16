"""Reproducible credit card exploration and customer-level spending regression."""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from insights import build_insights, filter_transactions

ROOT = Path(__file__).resolve().parent
REQUIRED = ["transaction_id", "customer_id", "date", "amount", "age", "gender", "city", "category", "card_type", "annual_income"]
NUMERIC = ["age", "annual_income"]
CATEGORICAL = ["gender", "city", "card_type"]


def generate_data(path: Path, customers: int = 800, seed: int = 42) -> None:
    """Create a fictional INR dataset for calendar 2024, including dirty rows."""
    rng = np.random.default_rng(seed)
    cities = ["Mumbai", "Delhi", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Kolkata", "Jaipur"]
    categories = ["Groceries", "Dining", "Shopping", "Travel", "Fuel", "Entertainment", "Utilities"]
    records = []
    for i in range(customers):
        age = int(rng.integers(21, 71))
        income = int(rng.lognormal(13.65, .48) // 1000 * 1000)
        city = str(rng.choice(cities))
        gender = str(rng.choice(["Female", "Male", "Non-binary"], p=[.48, .48, .04]))
        card = str(rng.choice(["Classic", "Gold", "Platinum"], p=[.45, .35, .20]))
        for _ in range(int(rng.integers(36, 85))):
            day = int(rng.integers(0, 366))
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day)
            category = str(rng.choice(categories))
            multiplier = {"Travel": 2.5, "Shopping": 1.6, "Utilities": 1.2}.get(category, 1)
            amount = round(float(rng.lognormal(6.5, .65) * (income / 850000) ** .65 * multiplier * (1.15 if date.month in [10, 11, 12] else 1)), 2)
            records.append([f"TX{len(records):07d}", f"C{i:05d}", date.strftime("%Y-%m-%d"), amount, age, gender, city, category, card, income])
    frame = pd.DataFrame(records, columns=REQUIRED)
    frame.loc[0, "date"] = "invalid-date"
    frame.loc[1, "amount"] = -50
    frame.loc[2, "age"] = 150
    frame.loc[3:10, "category"] = None
    frame.loc[11:20, "city"] = "  mumbai  "
    frame = pd.concat([frame, frame.iloc[30:55]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = raw[REQUIRED].copy()
    audit = {"input_rows": len(df)}
    # Exact duplicates can safely be removed; conflicting transaction IDs are rejected.
    df = df.drop_duplicates()
    audit["exact_duplicates_removed"] = len(raw) - len(df)
    for column in ["transaction_id", "customer_id", *CATEGORICAL, "category"]:
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)
    for column in [*CATEGORICAL, "category"]:
        df[column] = df[column].str.title()
    audit["categorical_cells_filled"] = int(df[[*CATEGORICAL, "category"]].isna().sum().sum())
    df[[*CATEGORICAL, "category"]] = df[[*CATEGORICAL, "category"]].fillna("Unknown")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="ISO8601", utc=True).dt.tz_convert(None)
    for column in ["amount", *NUMERIC]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    valid = (df["date"].notna() & df["amount"].gt(0) & df["age"].between(18, 100)
             & df["annual_income"].gt(0) & np.isfinite(df[["amount", *NUMERIC]]).all(axis=1)
             & df["transaction_id"].notna() & df["customer_id"].notna())
    audit["invalid_rows_removed"] = int((~valid).sum())
    df = df.loc[valid].copy()
    conflicts = df["transaction_id"].duplicated(keep=False)
    audit["conflicting_id_rows_removed"] = int(conflicts.sum())
    df = df.loc[~conflicts].sort_values(["date", "transaction_id"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid transactions remain after cleaning.")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["age_group"] = pd.cut(df["age"], [17, 24, 34, 44, 54, 64, 100], labels=["18–24", "25–34", "35–44", "45–54", "55–64", "65+"])
    audit["clean_rows"] = len(df)
    return df, audit


def summarize(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def segment(column):
        result = df.groupby(column, observed=True).agg(total_spend=("amount", "sum"), transactions=("amount", "size"), customers=("customer_id", "nunique"), average_transaction=("amount", "mean"))
        result["spend_per_customer"] = result["total_spend"] / result["customers"]
        return result.sort_values("total_spend", ascending=False)
    return {column: segment(column) for column in ["city", "gender", "age_group", "category", "card_type", "month"]}


def customer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Customers must have a stable profile: exclude ambiguous customers from modeling.
    stable = df.groupby("customer_id")[[*NUMERIC, *CATEGORICAL]].nunique().le(1).all(axis=1)
    profiles = df.groupby("customer_id")[[*NUMERIC, *CATEGORICAL]].first().loc[stable]
    months = len(pd.period_range(df["date"].min().to_period("M"), df["date"].max().to_period("M"), freq="M"))
    profiles["monthly_spend"] = df.groupby("customer_id")["amount"].sum() / months
    return profiles


def encode(frame: pd.DataFrame, metadata: dict | None = None):
    numeric = frame[NUMERIC].astype(float)
    if metadata is None:
        metadata = {"mean": numeric.mean().to_dict(), "std": numeric.std().replace(0, 1).fillna(1).to_dict()}
    numeric = (numeric - pd.Series(metadata["mean"])) / pd.Series(metadata["std"])
    categories = pd.get_dummies(frame[CATEGORICAL], dtype=float)
    features = pd.concat([numeric, categories], axis=1)
    if "columns" not in metadata:
        metadata["columns"] = features.columns.tolist()
    return features.reindex(columns=metadata["columns"], fill_value=0).to_numpy(dtype="float32"), metadata


def train_model(df: pd.DataFrame, output: Path, epochs: int, seed: int) -> dict:
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    profiles = customer_features(df)
    if len(profiles) < 50:
        raise ValueError("Model training needs at least 50 customers with consistent profiles. Use --skip-model for analysis only.")
    profiles = profiles.iloc[np.random.default_rng(seed).permutation(len(profiles))]
    n = len(profiles)
    train, val, test = np.split(np.arange(n), [int(n * .70), int(n * .85)])
    x_train, metadata = encode(profiles.iloc[train])
    x_val, _ = encode(profiles.iloc[val], metadata)
    x_test, _ = encode(profiles.iloc[test], metadata)
    target = profiles["monthly_spend"].to_numpy(dtype="float32")
    scale = float(target[train].mean())
    model = tf.keras.Sequential([tf.keras.Input(shape=(x_train.shape[1],)), tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(16, activation="relu"), tf.keras.layers.Dense(1)])
    model.compile(optimizer=tf.keras.optimizers.Adam(.003), loss="mse")
    # Explicit datasets bound resource use on local computers.
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, target[train] / scale)).shuffle(len(train), seed=seed).batch(64).with_options(options)
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, target[val] / scale)).batch(64).with_options(options)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True)])
    predictions = np.maximum(model(x_test, training=False).numpy().ravel() * scale, 0)
    actual = target[test]
    baseline = np.full_like(actual, target[train].mean())
    metrics = {"train_customers": len(train), "validation_customers": len(val), "test_customers": len(test), "excluded_inconsistent_customers": int(df.customer_id.nunique() - n), "epochs_run": len(history.history["loss"]), "mae_inr": float(np.mean(abs(actual - predictions))), "rmse_inr": float(np.sqrt(np.mean((actual - predictions) ** 2))), "baseline_mae_inr": float(np.mean(abs(actual - baseline))), "r_squared": float(1 - np.sum((actual - predictions) ** 2) / np.sum((actual - actual.mean()) ** 2)) if np.var(actual) else None}
    model.save(output / "spending_model.keras")
    metadata["target_scale"] = scale
    (output / "preprocessing.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"customer_id": profiles.iloc[test].index, "actual_monthly_spend": actual, "predicted_monthly_spend": predictions, "baseline_prediction": baseline}).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(history.history).to_csv(output / "training_history.csv", index=False)
    return metrics


def create_report(df, audit, tables, output, metrics, synthetic, monthly_budget=None, filters=None):
    plt.rcParams.update({"figure.facecolor": "#111c30", "axes.facecolor": "#111c30", "text.color": "#e2e8f0", "axes.labelcolor": "#a8b5ca", "xtick.color": "#a8b5ca", "ytick.color": "#a8b5ca", "axes.edgecolor": "#344155", "font.size": 10})
    charts = []
    for key, title, measure in [("month", "Monthly spending", "total_spend"), ("category", "Where the money goes", "total_spend"), ("city", "Spending by city", "spend_per_customer"), ("age_group", "Spending by age", "spend_per_customer"), ("gender", "Spending by gender", "spend_per_customer"), ("card_type", "Spending by card tier", "spend_per_customer")]:
        series = tables[key][measure]
        if key in ["month", "age_group"]:
            series = series.sort_index()
        fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
        if key == "month":
            ax.plot(series.index, series.values, color="#60d9bc", marker="o", linewidth=2)
            ax.fill_between(series.index, series.values, alpha=.12, color="#60d9bc")
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.barh(series.index.astype(str), series.values, color="#60d9bc", height=.6)
            ax.invert_yaxis()
        formatter = FuncFormatter(lambda x, _: f"{x / 100000:,.1f}L" if abs(x) >= 100000 else f"{x / 1000:,.1f}k")
        (ax.yaxis if key == "month" else ax.xaxis).set_major_formatter(formatter)
        ax.set_title(title, loc="left", fontweight="bold", pad=18)
        ax.set_ylabel("INR" if key == "month" else "")
        ax.set_xlabel("" if key == "month" else ("INR per customer · full observation window" if measure == "spend_per_customer" else "INR · total spend"))
        ax.spines[["top", "right"]].set_visible(False)
        path = output / f"{key}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        encoded = base64.b64encode(path.read_bytes()).decode()
        charts.append(f'<figure><img alt="{title}" src="data:image/png;base64,{encoded}"></figure>')
    if metrics:
        model_text = f"Held-out MAE: INR {metrics['mae_inr']:,.0f} · Mean-baseline MAE: INR {metrics['baseline_mae_inr']:,.0f} · Test customers: {metrics['test_customers']}"
    else:
        model_text = "Model training was skipped. Run without --skip-model to train and evaluate TensorFlow."
    top = tables["category"].iloc[0]
    share = top.total_spend / df.amount.sum() * 100
    title = "Synthetic demonstration" if synthetic else "User-supplied dataset · assumed INR"
    cards = [("Total spend", f"INR {df.amount.sum():,.0f}"), ("Transactions", f"{len(df):,}"), ("Customers", f"{df.customer_id.nunique():,}"), ("Average transaction", f"INR {df.amount.mean():,.0f}")]
    body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Credit Card Usage Analysis</title><style>
    *{{box-sizing:border-box}}body{{margin:0;background:#09111f;color:#e2e8f0;font:15px/1.6 system-ui,sans-serif}}main{{max-width:1180px;margin:auto;padding:48px 24px}}header{{border-bottom:1px solid #273349;padding-bottom:24px}}.eyebrow{{color:#60d9bc;letter-spacing:2px;font-size:12px;text-transform:uppercase}}h1{{font-size:38px;letter-spacing:-1px;margin:12px 0}}p{{color:#a8b5ca}}.cards,.charts{{display:grid;gap:16px;margin:28px 0}}.cards{{grid-template-columns:repeat(4,1fr)}}.cards article,section,figure{{background:#111c30;border:1px solid #273349;border-radius:14px;padding:20px;margin:0}}small{{color:#a8b5ca}}strong{{display:block;font-size:23px;margin-top:8px}}.charts{{grid-template-columns:repeat(2,1fr)}}figure{{padding:8px}}img{{width:100%;display:block}}section{{margin:22px 0}}pre{{white-space:pre-wrap;color:#a8b5ca}}@media(max-width:750px){{.cards,.charts{{grid-template-columns:1fr}}h1{{font-size:30px}}}}</style><main>
    <header><div class="eyebrow">Spending intelligence / {title}</div><h1>Credit Card Usage Analysis</h1><p>{df.date.min():%d %b %Y} – {df.date.max():%d %b %Y} · Demographics, geography, and spending behavior</p></header>
    <div class="cards">{''.join(f'<article><small>{label}</small><strong>{value}</strong></article>' for label, value in cards)}</div>
    <section><h2>Portfolio snapshot</h2><p>{html.escape(str(tables['category'].index[0]))} accounts for {share:.1f}% of recorded spending. Segment charts report spend per observed customer so differences in segment size are easier to compare. Individual customers may appear in multiple segments when their attributes vary.</p></section>
    <div class="charts">{''.join(charts)}</div><section><h2>TensorFlow spending model</h2><p>{model_text}</p><p>Target: average monthly recorded spending over the dataset window, including inactive months. Customers are split 70/15/15 into training, validation, and test sets. Preprocessing is fitted on training customers only. This estimates spending from customer profiles; it is not a future spending forecast or creditworthiness assessment.</p></section>
    <section><h2>Data quality audit</h2><pre>{html.escape(json.dumps(audit, indent=2))}</pre><p>Missing segment labels become Unknown. Invalid values and conflicting transaction IDs are removed. High positive amounts are retained because they may represent valid purchases.</p></section>
    <p>{'All transactions are fictional. Patterns reflect the data generator and do not establish real consumer behavior.' if synthetic else 'Results describe the supplied observations. Currency is assumed to be INR; convert mixed currencies before importing.'} Customer coverage and segment sizes affect comparisons; associations are descriptive.</p></main></html>'''
    extra = build_insights(df, output, monthly_budget, filters, metrics)
    body = body.replace('<section><h2>Data quality audit</h2>', extra + '<section><h2>Data quality audit</h2>')
    (output / "report.html").write_text(body, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV with the documented transaction schema")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", help="Inclusive start date, YYYY-MM-DD")
    parser.add_argument("--end", help="Inclusive end date, YYYY-MM-DD")
    parser.add_argument("--city", action="append", help="City filter; repeat to select multiple cities")
    parser.add_argument("--category", action="append", help="Category filter; repeat to select multiple categories")
    parser.add_argument("--monthly-budget", type=float, help="Positive monthly portfolio budget in INR")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.monthly_budget is not None and (not np.isfinite(args.monthly_budget) or args.monthly_budget <= 0):
        parser.error("--monthly-budget must be positive and finite")
    try:
        for value in [args.start, args.end]:
            if value is not None:
                from datetime import datetime
                datetime.strptime(value, "%Y-%m-%d")
        if args.start and args.end and pd.Timestamp(args.start) > pd.Timestamp(args.end):
            parser.error("--start must be on or before --end")
    except ValueError:
        parser.error("Dates must be valid YYYY-MM-DD values")
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.input
    if source is None:
        source = args.output / "synthetic_transactions.csv"
        generate_data(source, seed=args.seed)
    df, audit = clean_data(pd.read_csv(source))
    cleaned_rows = len(df)
    df = filter_transactions(df, args.start, args.end, args.city, args.category)
    audit["filtered_out_rows"] = cleaned_rows - len(df)
    audit["analysis_rows"] = len(df)
    filters = {key: value for key, value in {"start": args.start, "end": args.end, "cities": args.city, "categories": args.category}.items() if value is not None}
    df.to_csv(args.output / "clean_transactions.csv", index=False)
    (args.output / "quality_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    tables = summarize(df)
    for key, table in tables.items():
        table.to_csv(args.output / f"summary_{key}.csv")
    # Do not leave prior model artifacts that could be mistaken for this run's outputs.
    for name in ["spending_model.keras", "preprocessing.json", "model_metrics.json", "test_predictions.csv", "training_history.csv", "model_errors_by_city.csv", "model_diagnostics.png"]:
        (args.output / name).unlink(missing_ok=True)
    metrics = None if args.skip_model else train_model(df, args.output, args.epochs, args.seed)
    create_report(df, audit, tables, args.output, metrics, args.input is None, args.monthly_budget, filters)
    print(json.dumps({"report": str((args.output / 'report.html').resolve()), "audit": audit, "model": metrics}, indent=2))


if __name__ == "__main__":
    main()
