import unittest

import numpy as np
import pandas as pd

from analysis import clean_data, customer_features, encode, summarize
from insights import monthly_trends, customer_segments, unusual_transactions, weekday_patterns, filter_transactions


def sample():
    return pd.DataFrame([
        dict(transaction_id="T1", customer_id="C1", date="2024-01-01", amount=100, age=30, gender=" female ", city=" mumbai ", category="Dining", card_type="Gold", annual_income=800000),
        dict(transaction_id="T2", customer_id="C1", date="2024-03-01", amount=200, age=30, gender="Female", city="Mumbai", category=None, card_type="Gold", annual_income=800000),
        dict(transaction_id="T3", customer_id="C2", date="2024-03-03", amount=600, age=50, gender="Male", city="Delhi", category="Travel", card_type="Classic", annual_income=900000),
    ])


class AnalysisTests(unittest.TestCase):
    def test_cleaning_audit_and_conflicting_ids(self):
        raw = sample()
        duplicate = raw.iloc[[0]]
        bad = raw.iloc[[1]].assign(transaction_id="BAD", amount=float("inf"))
        conflict = raw.iloc[[2]].assign(amount=700)
        clean, audit = clean_data(pd.concat([raw, duplicate, bad, conflict]))
        self.assertEqual(audit["exact_duplicates_removed"], 1)
        self.assertEqual(audit["invalid_rows_removed"], 1)
        self.assertEqual(audit["conflicting_id_rows_removed"], 2)
        self.assertEqual(clean.city.tolist(), ["Mumbai", "Mumbai"])
        self.assertEqual(clean.category.iloc[1], "Unknown")
        self.assertEqual(audit["input_rows"], audit["clean_rows"] + sum(audit[k] for k in ["exact_duplicates_removed", "invalid_rows_removed", "conflicting_id_rows_removed"]))

    def test_segments_and_inactive_months(self):
        clean, _ = clean_data(sample())
        summary = summarize(clean)
        self.assertEqual(summary["city"].loc["Mumbai", "spend_per_customer"], 300)
        self.assertEqual(summary["city"].total_spend.sum(), clean.amount.sum())
        profiles = customer_features(clean)
        self.assertEqual(profiles.loc["C1", "monthly_spend"], 100)
        self.assertEqual(profiles.loc["C2", "monthly_spend"], 200)

    def test_preprocessing_fits_training_only(self):
        frame, _ = clean_data(sample())
        train, metadata = encode(frame.iloc[:2])
        before = dict(metadata["mean"])
        test, _ = encode(frame.iloc[2:], metadata)
        self.assertEqual(before, metadata["mean"])
        self.assertEqual(train.shape[1], test.shape[1])
        self.assertFalse(any("Delhi" in c for c in metadata["columns"]))
        self.assertTrue(np.isfinite(test).all())

    def test_inconsistent_profiles_excluded(self):
        raw = sample()
        raw.loc[1, "age"] = 31
        clean, _ = clean_data(raw)
        self.assertEqual(customer_features(clean).index.tolist(), ["C2"])

    def test_invalid_schema_and_empty_data(self):
        with self.assertRaisesRegex(ValueError, "Missing required"):
            clean_data(pd.DataFrame({"amount": [1]}))
        with self.assertRaisesRegex(ValueError, "No valid"):
            clean_data(sample().assign(age=10))

    def test_growth_missing_months_and_zero_denominators(self):
        frame, _ = clean_data(sample())
        trends = monthly_trends(frame)
        self.assertEqual(trends.total_spend.tolist(), [100, 0, 800])
        self.assertEqual(trends.change_percent.iloc[1], -100)
        self.assertTrue(pd.isna(trends.change_percent.iloc[2]))
        self.assertEqual(trends.rolling_3_month_spend.iloc[2], 300)
        self.assertTrue(trends.partial_month.iloc[2])

    def test_rfm_ties_and_reference_date(self):
        frame, _ = clean_data(sample())
        segments = customer_segments(frame)
        self.assertEqual(segments.loc["C2", "recency_days"], 1)
        self.assertEqual(segments.loc["C1", "frequency"], 2)
        self.assertEqual(segments.monetary.sum(), frame.amount.sum())
        tied = frame.iloc[[0]].copy()
        result = customer_segments(tied)
        self.assertTrue(result.filter(like="_score").eq(3).all().all())

    def test_unusual_amounts_and_minimum_sample(self):
        frame, _ = clean_data(sample())
        group = pd.concat([frame.iloc[[0]]] * 10, ignore_index=True)
        group.loc[9, "amount"] = 10000
        result = unusual_transactions(group)
        self.assertEqual(result.unusual_amount.sum(), 1)
        self.assertEqual(result.iloc[9].category_upper_fence, 100)
        self.assertFalse(unusual_transactions(group.iloc[:7]).eligible.any())

    def test_weekdays_include_days_with_no_purchases(self):
        frame, _ = clean_data(sample())
        result = weekday_patterns(frame)
        self.assertEqual(result.calendar_days.sum(), 63)
        self.assertEqual(result.loc["Monday", "spend_per_calendar_day"], 100 / 9)
        self.assertEqual(result.total_spend.sum(), frame.amount.sum())

    def test_inclusive_filters_and_empty_scope(self):
        frame, _ = clean_data(sample())
        selected = filter_transactions(frame, "2024-03-01", "2024-03-01", [" mumbai "])
        self.assertEqual(selected.transaction_id.tolist(), ["T2"])
        self.assertEqual(filter_transactions(frame, categories=["travel"]).customer_id.tolist(), ["C2"])
        with self.assertRaisesRegex(ValueError, "No transactions"):
            filter_transactions(frame, cities=["Missing city"])


if __name__ == "__main__":
    unittest.main()
