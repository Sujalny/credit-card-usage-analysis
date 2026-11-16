"""Behavioral insights and report sections; no additional dependencies required."""
from __future__ import annotations

import base64
import html
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def monthly_trends(df):
    months = pd.period_range(df.date.min().to_period("M"), df.date.max().to_period("M"), freq="M")
    result = df.groupby(df.date.dt.to_period("M")).agg(total_spend=("amount", "sum"), transactions=("amount", "size"))
    result = result.reindex(months, fill_value=0)
    result.index.name = "month"
    previous = result.total_spend.shift()
    result["change_percent"] = (result.total_spend / previous.where(previous.ne(0)) - 1) * 100
    result["rolling_3_month_spend"] = result.total_spend.rolling(3, min_periods=1).mean()
    result["partial_month"] = [(m.start_time.normalize() < df.date.min().normalize() or m.end_time.normalize() > df.date.max().normalize()) for m in months]
    return result


def customer_segments(df):
    reference = df.date.max().normalize() + pd.Timedelta(days=1)
    result = df.groupby("customer_id").agg(last_purchase=("date", "max"), frequency=("transaction_id", "size"), monetary=("amount", "sum"))
    result["recency_days"] = (reference - result.last_purchase.dt.normalize()).dt.days
    # Tied values receive equal scores; all-equal cohorts receive a neutral score.
    for name, ascending in [("recency_days", False), ("frequency", True), ("monetary", True)]:
        values = result[name]
        score = np.ceil(values.rank(method="average", pct=True, ascending=ascending) * 5).clip(1, 5).astype(int)
        result[name + "_score"] = score if values.nunique() > 1 else 3
    r, f, m = (result[x + "_score"] for x in ["recency_days", "frequency", "monetary"])
    result["segment"] = np.select([(r >= 4) & (f >= 4) & (m >= 4), (r <= 2) & (m >= 4), f >= 4, r >= 4, r <= 2], ["Champions", "High-value inactive", "Frequent customers", "Recently active", "Inactive"], default="Developing")
    return result.sort_values("monetary", ascending=False)


def unusual_transactions(df, minimum_group=8):
    result = df.copy()
    grouped = result.groupby("category").amount
    q1 = grouped.transform(lambda s: s.quantile(.25))
    q3 = grouped.transform(lambda s: s.quantile(.75))
    result["category_upper_fence"] = q3 + 3 * (q3 - q1)
    result["comparison_count"] = grouped.transform("size")
    result["eligible"] = result.comparison_count.ge(minimum_group)
    result["unusual_amount"] = result.eligible & result.amount.gt(result.category_upper_fence)
    return result


def weekday_patterns(df):
    days = pd.date_range(df.date.min().normalize(), df.date.max().normalize())
    counts = pd.Series(days.dayofweek).value_counts().reindex(range(7), fill_value=0)
    result = df.groupby(df.date.dt.dayofweek).agg(total_spend=("amount", "sum"), transactions=("amount", "size")).reindex(range(7), fill_value=0)
    result["calendar_days"] = counts
    result["spend_per_calendar_day"] = result.total_spend / result.calendar_days.replace(0, np.nan)
    result.index = pd.Index(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], name="weekday")
    return result


def filter_transactions(df, start=None, end=None, cities=None, categories=None):
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df.date.ge(pd.Timestamp(start))
    if end is not None:
        mask &= df.date.lt(pd.Timestamp(end) + pd.Timedelta(days=1))
    if cities:
        mask &= df.city.isin([c.strip().title() for c in cities])
    if categories:
        mask &= df.category.isin([c.strip().title() for c in categories])
    result = df.loc[mask].copy()
    if result.empty:
        raise ValueError("No transactions match the selected filters.")
    return result


def build_insights(df, output, monthly_budget=None, filters=None, metrics=None):
    trends = monthly_trends(df)
    customers = customer_segments(df)
    flagged = unusual_transactions(df)
    weekdays = weekday_patterns(df)
    segments = customers.groupby("segment").agg(customers=("monetary", "size"), total_spend=("monetary", "sum"))
    segments["spend_share_percent"] = segments.total_spend / segments.total_spend.sum() * 100
    if monthly_budget is not None:
        if not math.isfinite(monthly_budget) or monthly_budget <= 0:
            raise ValueError("Monthly budget must be positive and finite.")
        trends["budget"] = monthly_budget
        trends["budget_used_percent"] = trends.total_spend / monthly_budget * 100
        trends["over_budget"] = trends.total_spend.gt(monthly_budget)
    tables = {"monthly_trends": trends, "customer_segments": customers, "segment_summary": segments, "weekday_patterns": weekdays, "unusual_transactions": flagged.loc[flagged.unusual_amount]}
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=name != "unusual_transactions")

    charts = []
    def finish(fig, name, title):
        path = output / f"{name}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        encoded = base64.b64encode(path.read_bytes()).decode()
        charts.append(f'<figure><img alt="{title}" src="data:image/png;base64,{encoded}"></figure>')

    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    ax.plot(trends.index.astype(str), trends.total_spend, label="Monthly spend", color="#60d9bc", marker="o")
    ax.plot(trends.index.astype(str), trends.rolling_3_month_spend, label="3-month rolling mean", color="#f4c572")
    if monthly_budget is not None:
        ax.axhline(monthly_budget, label="Monthly budget", color="#fa8795", linestyle="--")
    ax.set_title("Spending trend and budget", loc="left")
    ax.set_ylabel("INR")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    finish(fig, "spending_trend", "Spending trend and budget")

    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    ax.barh(segments.index, segments.customers, color="#60d9bc")
    ax.set_title("Customer behavior segments", loc="left")
    ax.set_xlabel("Customers")
    finish(fig, "customer_behavior", "Customer behavior segments")

    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    ax.bar(weekdays.index.str[:3], weekdays.spend_per_calendar_day, color="#8fa8ff")
    ax.set_title("Spending by day of week", loc="left")
    ax.set_ylabel("INR per calendar day")
    finish(fig, "weekday_spending", "Spending by day of week")

    # A row-normalized cross-tab reveals each city's category mix.
    mix = pd.crosstab(df.city, df.category, values=df.amount, aggfunc="sum").fillna(0)
    mix = mix.div(mix.sum(axis=1), axis=0) * 100
    mix.to_csv(output / "city_category_mix.csv")
    chart_mix = mix.loc[df.groupby("city").amount.sum().nlargest(12).index]
    chart_mix = chart_mix[chart_mix.sum().nlargest(10).index]
    fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
    heat = ax.imshow(chart_mix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(chart_mix.columns)), chart_mix.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(chart_mix)), chart_mix.index, fontsize=8)
    for row in range(len(chart_mix)):
        for column in range(len(chart_mix.columns)):
            value = chart_mix.iloc[row, column]
            ax.text(column, row, f"{value:.0f}%", ha="center", va="center", fontsize=7, color="white" if value >= 55 else "#09111f")
    ax.set_title("Category share within each city", loc="left")
    fig.colorbar(heat, ax=ax, label="% of city spending")
    finish(fig, "city_category_heatmap", "Category share within each city")

    model_section = ""
    if metrics:
        predictions = pd.read_csv(output / "test_predictions.csv")
        predictions["absolute_error"] = abs(predictions.actual_monthly_spend - predictions.predicted_monthly_spend)
        profiles = df.groupby("customer_id")["city"].first()
        predictions["city"] = predictions.customer_id.map(profiles)
        errors = predictions.groupby("city").agg(test_customers=("customer_id", "size"), mae_inr=("absolute_error", "mean"))
        errors.to_csv(output / "model_errors_by_city.csv")
        history = pd.read_csv(output / "training_history.csv")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
        axes[0].scatter(predictions.actual_monthly_spend, predictions.predicted_monthly_spend, color="#60d9bc", alpha=.6, s=16)
        high = predictions[["actual_monthly_spend", "predicted_monthly_spend"]].max().max()
        axes[0].plot([0, high], [0, high], "--", color="#f4c572")
        axes[0].set(xlabel="Actual monthly spend · INR", ylabel="Predicted · INR", title="Held-out predictions")
        axes[1].plot(np.arange(1, len(history) + 1), history.loss, label="Training", color="#60d9bc")
        axes[1].plot(np.arange(1, len(history) + 1), history.val_loss, label="Validation", color="#8fa8ff")
        axes[1].set(xlabel="Epoch", ylabel="Scaled mean squared error", title="Learning curves")
        axes[1].legend()
        finish(fig, "model_diagnostics", "Held-out predictions and learning curves")
        model_section = '<h3>Model errors by city</h3><p>These are held-out test results. Small groups give unstable estimates.</p>' + errors.round(2).to_html(classes="data-table", border=0)

    def table(frame):
        return frame.to_html(index=False, border=0, classes="data-table", float_format=lambda x: f"{x:,.2f}", escape=True, na_rep="—")

    unusual = flagged.loc[flagged.unusual_amount].sort_values("amount", ascending=False)
    leaderboard = customers.reset_index()[["customer_id", "segment", "recency_days", "frequency", "monetary"]]
    preview = unusual[["transaction_id", "date", "city", "category", "amount", "category_upper_fence"]].head(25)
    preview["date"] = preview.date.dt.strftime("%Y-%m-%d")
    budget_text = f" {int(trends.over_budget.sum())} months exceeded the INR {monthly_budget:,.0f} monthly portfolio budget." if monthly_budget is not None else " Supply --monthly-budget to compare monthly portfolio spending against a budget."
    scope = html.escape(str(filters or "All cleaned transactions"))
    return f'''<style>.table-wrap{{overflow:auto;max-height:420px}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #273349}}th{{color:#60d9bc;position:sticky;top:0;background:#111c30}}input{{background:#09111f;color:#e2e8f0;border:1px solid #455570;border-radius:8px;padding:10px;width:min(100%,360px);margin:10px 0}}details{{margin:16px 0}}summary{{cursor:pointer;color:#60d9bc}}</style>
    <section><h2>Trends and behavioral insights</h2><p>Scope: {scope}. RFM segments use recency, transaction frequency, and total spending, with relative scores from 1 to 5. Ties receive equal scores. These describe recorded activity within the selected window.</p><p>{len(unusual):,} unusually large transactions flagged among {int(flagged.eligible.sum()):,} eligible purchases.{budget_text}</p><p>Monthly changes are undefined after a zero-spend month. First and last months may be partial. Missing interior months count as zero. Weekday averages include calendar days with no transactions.</p></section>
    <div class="charts">{''.join(charts)}</div>
    <section><h2>Explore the results</h2><label for="customer-search">Search top 100 customers by ID or segment</label><br><input id="customer-search" type="search" placeholder="e.g. Champions or C00042"><div class="table-wrap" id="customer-results">{table(leaderboard.head(100))}</div><p>Sorted by total spending. Full customer scores and segments are in customer_segments.csv.</p>
    <details><summary>Monthly growth and budget details</summary><div class="table-wrap">{table(trends.reset_index())}</div></details>
    <details><summary>Top 25 unusual purchases</summary><p>Amounts exceeding category Q3 + 3 × IQR, using categories with at least 8 transactions. This flags statistical outliers for review; it does not identify fraud. Flagged transactions remain in all analyses.</p><div class="table-wrap">{table(preview)}</div></details>
    <details><summary>Held-out model evaluation by city</summary><div class="table-wrap">{model_section or '<p>Model training skipped.</p>'}</div></details></section>
    <script>document.getElementById('customer-search').addEventListener('input',function(){{const term=this.value.toLowerCase();document.querySelectorAll('#customer-results tbody tr').forEach(row=>{{row.hidden=!row.textContent.toLowerCase().includes(term)}})}});</script>'''
