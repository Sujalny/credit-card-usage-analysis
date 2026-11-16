# Credit Card Usage Analysis

A reproducible Python project using **Pandas, Matplotlib, and TensorFlow** to clean credit card transactions, compare spending across demographic and geographic segments, and estimate average monthly customer spending. Generates a standalone HTML report with eleven charts, CSV summaries, a data quality audit, and an evaluated neural network.

The default dataset is **synthetic**, covers calendar 2024, and uses INR. It contains fictional customers and deliberately injected quality issues. Findings from this dataset demonstrate the workflow; they are not claims about real consumers.

## Run

Use Python 3.11–3.13 in an isolated environment:

```powershell
git clone https://github.com/Sujalny/credit-card-usage-analysis.git
cd credit-card-usage-analysis
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe analysis.py
Start-Process outputs\report.html
```

If `py` is unavailable, use your Python executable in its place.

Analysis without model training (TensorFlow is only imported when training):

```powershell
.\.venv\Scripts\python.exe analysis.py --skip-model
```

Each run replaces the named outputs in its output directory. Use a separate directory to preserve a prior run:

```powershell
.\.venv\Scripts\python.exe analysis.py --input data\transactions.csv --output outputs\my-dataset --epochs 100 --seed 42
```

## Additional features

- **Monthly trends:** month-over-month growth, three-month rolling spending, zero-filled interior months, and partial-month indicators.
- **Budget tracking:** optional monthly portfolio budget, utilization percentage, and over-budget flags.
- **Customer segmentation:** recency, frequency, and monetary (RFM) scores with Champions, High-value inactive, Frequent customers, Recently active, Inactive, and Developing segments. Scores are relative to the selected customer population; ties receive equal scores and constant features receive a neutral score of 3. Labels describe observed activity, not predicted churn.
- **Unusual purchases:** transactions above their category's Q3 + 3 × IQR, with at least eight comparison transactions. These statistical flags do not identify fraud; flagged transactions remain in all summaries.
- **Weekday behavior:** spending per calendar day, including dates with no purchases.
- **Geographic category mix:** a heatmap of within-city category spending shares. Chart displays up to the top 12 cities and 10 categories; the CSV includes all groups.
- **Model diagnostics:** held-out actual versus predicted spending, learning curves, and city-level test errors with sample counts.
- **Searchable customer table:** search the top 100 customers by ID or segment directly in the offline HTML report. Expand report sections to inspect growth, unusual purchases, and model evaluation.

Filter by date, city, and category and track a budget:

```powershell
.\.venv\Scripts\python.exe analysis.py --start 2024-10-01 --end 2024-12-31 --city Mumbai --city Delhi --monthly-budget 1000000 --output outputs\quarterly
.\.venv\Scripts\python.exe analysis.py --category Dining --category Travel --skip-model --output outputs\categories
```

Dates are inclusive. Repeat `--city` or `--category` to select multiple values; matches are case-insensitive. All analyses, RFM scores, training targets, and budget checks use only the selected transactions. Budgets apply to the selected portfolio, not individual customers, and are not prorated for partial months. `quality_audit.json` distinguishes all cleaned rows from filtered analysis rows; the exported `clean_transactions.csv` contains the selected scope. Using `--input` labels the report as user-supplied data, so retain the synthetic-data context if importing generated demonstration data.

Additional exports are `monthly_trends.csv`, `customer_segments.csv`, `segment_summary.csv`, `unusual_transactions.csv`, `weekday_patterns.csv`, `city_category_mix.csv`, and (with training) `model_errors_by_city.csv`. Four additional analytical charts and one model diagnostic chart are embedded in the report. Run the existing test command to check filtering, zero-month growth, RFM ties, anomaly thresholds, and calendar-day normalization alongside the original checks.

## Input schema

One purchase per row in a UTF-8 CSV, with these required headers:

| Column | Meaning / example |
| --- | --- |
| transaction_id | Unique purchase ID, `TX00001` |
| customer_id | Stable anonymized customer ID, `C00001` |
| date | ISO 8601 date, `2024-10-15` |
| amount | Positive purchase amount in INR, `1499.50` |
| age | Age 18–100, `29` |
| gender | Segment label, `Female` |
| city | Geographic segment, `Mumbai` |
| category | Merchant category, `Groceries` |
| card_type | Card tier, `Gold` |
| annual_income | Positive annual income in INR, `850000` |

Use one currency and a complete observation window. Refunds and nonpositive transactions are excluded; net spending is not calculated. Do not include card numbers or personal identifiers. Extra columns are ignored. Missing required headers and an empty cleaned dataset raise an error.

## Cleaning and exploration

- Remove exact duplicate rows; reject all remaining rows sharing a transaction ID rather than choose a conflicting record.
- Parse dates and numeric values; remove missing identifiers, invalid dates, nonfinite/nonpositive amounts or income, and ages outside 18–100.
- Trim and title-case segment labels; fill missing segment labels with `Unknown`.
- Retain large positive purchases. The pipeline does not assume they are errors.
- Export monthly, category, city, gender, age, and card-tier summaries with total spending, transaction count, distinct customers, average transaction value, and spending per customer.

Segment spending per customer is for the **entire observation window**. A customer with changing segment attributes can appear in more than one segment. Customer counts across segments therefore need not add up to the portfolio customer count. Plots are descriptive; group size, customer coverage, and confounding affect comparisons.

## TensorFlow model

A dense 32 → 16 → 1 regression model estimates customer average monthly recorded spending from age, income, gender, city, and card tier. Customers with inconsistent profiles are excluded from modeling and counted in the metrics; their transactions remain in the descriptive analysis.

The target is total recorded spending divided by every calendar month between the dataset's first and last month, including inactive months. For partial months, short customer histories, or incomplete data, this can understate monthly activity. The target describes the observed period, not future spending.

Customers are randomly split into 70% training, 15% validation, and 15% test partitions. Numeric scaling and one-hot feature columns are fitted on training data only. Unknown categories at prediction time use all-zero indicators for that feature. Early stopping uses validation loss; reported MAE, RMSE, and R² use only the held-out test customers. A training-mean baseline is evaluated on the same test set. At least 50 customers with consistent profiles are required to train. Seeds are fixed, though exact floating-point reproducibility can vary across hardware.

This is an educational spending model, not a credit scoring or lending model. Synthetic accuracy does not establish performance on real transaction data.

Predict with the saved model:

```powershell
.\.venv\Scripts\python.exe predict.py data\example_profiles.csv
# For a model trained into a custom output folder:
.\.venv\Scripts\python.exe predict.py data\example_profiles.csv --model-dir outputs\my-dataset --output outputs\my-dataset\predictions.csv
```

## Outputs

| File | Contents |
| --- | --- |
| `report.html` | Responsive offline report, with charts embedded |
| `synthetic_transactions.csv` | Generated raw data, default runs only |
| `clean_transactions.csv` | Cleaned transactions with month and age group |
| `quality_audit.json` | Row-removal and missing-category counts |
| `summary_*.csv` | Six segment aggregation tables |
| `*.png` | Eleven reusable Matplotlib charts when training runs |
| `spending_model.keras` | Trained TensorFlow model |
| `preprocessing.json` | Training feature order, means, deviations, target scale |
| `model_metrics.json` | Held-out errors, baseline error, and partition sizes |
| `test_predictions.csv` | Actual, model, and baseline values for test customers |
| `training_history.csv` | Training and validation loss per epoch |

Model files are generated only when training runs. Each analysis run removes earlier named model artifacts in its output folder to avoid confusing stale results with current results. Generated files and environments are excluded from version control.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Tests cover duplicate/conflict handling, audit reconciliation, missing categories, segment aggregation, inactive-month targets, inconsistent customer profiles, schema validation, and training-only preprocessing. The full pipeline and prediction commands also exercise training, serialization, inference, and report generation.
