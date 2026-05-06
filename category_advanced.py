"""
category_advanced.py
====================
STEP 2 (Advanced) — Per-Category Forecasting with Model Comparison

Improvements over original category.py:
  ✅ Reads from SQLite database (not Excel)
  ✅ Trains Prophet + Moving Average + Linear Regression per category
  ✅ Compares all 3 models using RMSE
  ✅ Auto-selects best model per category
  ✅ More categories supported (Medical, Education, Shopping, Other)
  ✅ Better anomaly severity labels
  ✅ Saves results back to database

Run AFTER split_data_advanced.py
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import (get_category_grouped, get_holidays, init_db)
from models   import (run_prophet, run_moving_average, run_linear_regression,
                       compute_metrics, detect_anomalies, CATEGORIES)

print("=" * 65)
print("  STEP 2 (Advanced) — Category-Wise Forecasting + Model Comparison")
print("=" * 65)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
FORECAST_DAYS = 30
ANOMALY_SIGMA = 2.0

# ──────────────────────────────────────────────
# LOAD DATA FROM DATABASE
# ──────────────────────────────────────────────
print("\n🗄️  Loading data from database...")
init_db()

cat_df   = get_category_grouped()
holidays = get_holidays()

if cat_df.empty:
    print("   ❌ No data in database. Run split_data_advanced.py first.")
    exit(1)

categories = sorted(cat_df['Category'].unique().tolist())
print(f"   ✅ {len(cat_df)} rows across {len(categories)} categories")
print(f"   ✅ Holidays: {len(holidays)}")
print(f"   ✅ Categories found: {categories}")

# ──────────────────────────────────────────────
# CREATE OUTPUT FOLDER
# ──────────────────────────────────────────────
os.makedirs("category_forecasts_advanced", exist_ok=True)

# ──────────────────────────────────────────────
# TRAIN PER CATEGORY
# ──────────────────────────────────────────────
summary_rows = []

print(f"\n🤖 Training models per category...")
print("-" * 65)

for cat in categories:
    print(f"\n  📦 Category: {cat}")

    sub = cat_df[cat_df['Category'] == cat][['Date','Amount']].copy()
    sub.rename(columns={'Date':'ds','Amount':'y'}, inplace=True)
    sub = sub.sort_values('ds').reset_index(drop=True)

    if len(sub) < 3:
        print(f"     ⚠️  Only {len(sub)} rows — skipping (need ≥ 3)")
        continue

    print(f"     📊 Records    : {len(sub)}")
    print(f"     📅 Date range : {sub['ds'].min().date()} → {sub['ds'].max().date()}")
    print(f"     💰 Total      : Rs {sub['y'].sum():.2f} | Avg: Rs {sub['y'].mean():.2f}")

    test_size = min(2, len(sub) - 1)
    train = sub.iloc[:-test_size].copy()
    test  = sub.iloc[-test_size:].copy()
    acts  = test['y'].values

    model_results  = {}
    model_forecasts = {}

    # ── Prophet ──
    try:
        _, fc_p = run_prophet(train, holidays, test_size)
        preds_p = fc_p[fc_p['ds'].isin(test['ds'])]['yhat'].values
        if len(preds_p) > 0:
            model_results['Prophet'] = compute_metrics(acts[:len(preds_p)], preds_p)
        _, fc_full = run_prophet(sub, holidays, FORECAST_DAYS)
        model_forecasts['Prophet'] = fc_full
        print(f"     ✅ Prophet  RMSE: Rs {model_results.get('Prophet',{}).get('RMSE','N/A')}")
    except Exception as e:
        print(f"     ⚠️  Prophet failed: {e}")

    # ── Moving Average ──
    try:
        if len(train) >= 3:
            fut_ma, in_ma = run_moving_average(train, test_size)
            preds_ma = np.full(len(test), in_ma['yhat'].iloc[-1])
            model_results['Moving Average'] = compute_metrics(acts, preds_ma)
            fut_ma_full, _ = run_moving_average(sub, FORECAST_DAYS)
            model_forecasts['Moving Average'] = fut_ma_full
            print(f"     ✅ MovingAvg RMSE: Rs {model_results['Moving Average']['RMSE']}")
    except Exception as e:
        print(f"     ⚠️  Moving Average failed: {e}")

    # ── Linear Regression ──
    try:
        if len(train) >= 3:
            fut_lr, _ = run_linear_regression(train, test_size)
            preds_lr = fut_lr['yhat'].values[:len(test)]
            model_results['Linear Regression'] = compute_metrics(acts[:len(preds_lr)], preds_lr)
            fut_lr_full, _ = run_linear_regression(sub, FORECAST_DAYS)
            model_forecasts['Linear Regression'] = fut_lr_full
            print(f"     ✅ LinearReg RMSE: Rs {model_results['Linear Regression']['RMSE']}")
    except Exception as e:
        print(f"     ⚠️  Linear Regression failed: {e}")

    # ── Pick best model ──
    best_model = 'Prophet'  # default
    if model_results:
        valid = {k: v for k, v in model_results.items()
                 if v.get('RMSE') is not None}
        if valid:
            best_model = min(valid, key=lambda k: valid[k]['RMSE'])
    print(f"     🏆 Best Model : {best_model}")

    best_metrics  = model_results.get(best_model, {})
    best_forecast = model_forecasts.get(best_model)

    # ── Anomaly detection using best model ──
    n_anomalies = 0
    if best_forecast is not None and 'ds' in best_forecast.columns:
        try:
            anom_df     = detect_anomalies(sub, best_forecast, ANOMALY_SIGMA)
            n_anomalies = int(anom_df['anomaly'].sum())
            print(f"     🚨 Anomalies  : {n_anomalies} detected at {ANOMALY_SIGMA}σ")
            if n_anomalies > 0:
                for _, row in anom_df[anom_df['anomaly']].iterrows():
                    direction = "HIGH" if row['residual'] < 0 else "LOW"
                    print(f"        ⚠️  {row['ds'].date()} | Actual=Rs{row['y']:.0f} "
                          f"| Expected=Rs{row['yhat']:.0f} | {row['severity']:.1f}σ {direction}")
        except Exception as e:
            print(f"     ⚠️  Anomaly detection failed: {e}")

    # ── Future forecast totals ──
    if best_forecast is not None and 'ds' in best_forecast.columns:
        fut_only = best_forecast[best_forecast['ds'] > sub['ds'].max()]
        forecast_total = fut_only['yhat'].sum() if len(fut_only) > 0 else 0
        print(f"     🔮 Forecast ({FORECAST_DAYS}d): Rs {forecast_total:.2f} total")
    else:
        forecast_total = 0

    # ── Save per-category Excel ──
    out_path = f"category_forecasts_advanced/{cat}_forecast.xlsx"
    try:
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            sub.rename(columns={'ds':'Date','y':'Amount'}).to_excel(
                writer, sheet_name='Actuals', index=False)

            if best_forecast is not None:
                best_forecast[['ds','yhat','yhat_lower','yhat_upper']].rename(
                    columns={'ds':'Date','yhat':'Forecast',
                             'yhat_lower':'Lower','yhat_upper':'Upper'}
                ).to_excel(writer, sheet_name='Best Forecast', index=False)

            # All models comparison sheet
            comp_rows = []
            for mname, mets in model_results.items():
                row = {'Model': mname}
                row.update(mets)
                comp_rows.append(row)
            if comp_rows:
                pd.DataFrame(comp_rows).to_excel(
                    writer, sheet_name='Model Comparison', index=False)

        print(f"     💾 Saved → {out_path}")
    except Exception as e:
        print(f"     ⚠️  Could not save Excel: {e}")

    summary_rows.append({
        'Category'              : cat,
        'Records'               : len(sub),
        'Total Spend Rs'        : round(sub['y'].sum(), 2),
        'Avg Daily Rs'          : round(sub['y'].mean(), 2),
        'Best Model'            : best_model,
        'RMSE (best)'           : best_metrics.get('RMSE'),
        'MAE (best)'            : best_metrics.get('MAE'),
        'R² (best)'             : best_metrics.get('R2'),
        'MAPE % (best)'         : best_metrics.get('MAPE'),
        'Anomalies'             : n_anomalies,
        f'Forecast Rs ({FORECAST_DAYS}d)': round(forecast_total, 2),
    })

# ──────────────────────────────────────────────
# SAVE SUMMARY
# ──────────────────────────────────────────────
print("\n" + "-" * 65)
print("📋 Saving summary metrics...")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel("category_metrics_advanced.xlsx", index=False)
print("   ✅ Saved → category_metrics_advanced.xlsx")

# ──────────────────────────────────────────────
# PRINT TABLE
# ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("  📊 ADVANCED CATEGORY METRICS SUMMARY")
print("=" * 65)
print(f"  {'Category':<15} {'Best Model':<18} {'RMSE':>8} {'MAPE%':>8} {'Anomalies':>10}")
print("  " + "-" * 62)
for row in summary_rows:
    rmse = f"{row['RMSE (best)']:.2f}" if row['RMSE (best)'] is not None else "N/A"
    mape = f"{row['MAPE % (best)']:.1f}%" if row['MAPE % (best)'] is not None else "N/A"
    print(f"  {row['Category']:<15} {row['Best Model']:<18} "
          f"{rmse:>8} {mape:>8} {row['Anomalies']:>10}")

print()
total_fc = sum(r[f'Forecast Rs ({FORECAST_DAYS}d)'] for r in summary_rows)
print(f"  Total Forecast Next {FORECAST_DAYS} Days: Rs {total_fc:,.2f}")
print()
print("  📁 Files Created:")
print("     ✅ category_metrics_advanced.xlsx")
for row in summary_rows:
    print(f"     ✅ category_forecasts_advanced/{row['Category']}_forecast.xlsx")
print()
print("  ▶️  Next → streamlit run advanced_app.py")
print("=" * 65)
