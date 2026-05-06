"""
prophet_advanced.py
===================
STEP 3 (Advanced) — Overall Forecasting with Full Pipeline

Improvements over original prophet_model.py:
  ✅ Reads from SQLite database
  ✅ Hyperparameter tuning (24 combinations)
  ✅ Walk-forward cross-validation
  ✅ Model comparison (Prophet vs MA vs LR)
  ✅ Weekly + Monthly summary
  ✅ Anomaly detection with severity labels
  ✅ Spike detection for future days
  ✅ Saves 8-sheet Excel report

Run AFTER split_data_advanced.py and category_advanced.py
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import get_daily_totals, get_holidays, init_db
from models   import (run_prophet, hyperparameter_tuning, cross_validate,
                       compare_all_models, detect_anomalies, compute_metrics,
                       run_moving_average, run_linear_regression)

print("=" * 65)
print("  STEP 3 (Advanced) — Overall Prophet Forecasting")
print("=" * 65)

FORECAST_DAYS = 30
ANOMALY_SIGMA = 2.0
SPIKE_MULT    = 1.5    # flag days > 1.5x average

# ──────────────────────────────────────────────
# LOAD FROM DATABASE
# ──────────────────────────────────────────────
print("\n🗄️  Loading daily totals from database...")
init_db()

daily_df = get_daily_totals()
holidays = get_holidays()

if len(daily_df) < 3:
    print("   ❌ Not enough data (need ≥ 3 days). Run split_data_advanced.py first.")
    exit(1)

print(f"   ✅ {len(daily_df)} daily records loaded")
print(f"   ✅ Date range : {daily_df['ds'].min().date()} → {daily_df['ds'].max().date()}")
print(f"   ✅ Total spend: Rs {daily_df['y'].sum():,.2f}")
print(f"   ✅ Avg daily  : Rs {daily_df['y'].mean():.2f}")
print(f"   ✅ Holidays   : {len(holidays)}")

# ──────────────────────────────────────────────
# PHASE A — HYPERPARAMETER TUNING
# ──────────────────────────────────────────────
print("\n" + "-" * 65)
print("🎯 PHASE A — Hyperparameter Grid Search (24 combinations)...")

tune_df, best_cp, best_sp, best_mode = hyperparameter_tuning(daily_df, holidays)

if not tune_df.empty:
    print(f"\n   Top 5 Combinations:")
    print(f"   {'CP':>6} {'SP':>6} {'Mode':>14} {'RMSE':>10} {'MAPE%':>8}")
    print("   " + "-" * 48)
    for _, row in tune_df.head(5).iterrows():
        print(f"   {row['changepoint_prior_scale']:>6} "
              f"{row['seasonality_prior_scale']:>6} "
              f"{row['seasonality_mode']:>14} "
              f"Rs {row['RMSE']:>7.2f} "
              f"{row['MAPE']:>7.1f}%")
    print(f"\n   🏆 Best: CP={best_cp} | SP={best_sp} | Mode={best_mode}")
else:
    print("   ℹ️  Tuning skipped (not enough data for split) — using defaults")

# ──────────────────────────────────────────────
# PHASE B — CROSS VALIDATION
# ──────────────────────────────────────────────
print("\n" + "-" * 65)
print("📊 PHASE B — Walk-Forward Cross Validation (3 folds)...")

cv_results = cross_validate(daily_df, holidays, best_cp, best_sp, best_mode, n_folds=3)

if cv_results:
    print(f"\n   {'Fold':>5} {'Train':>7} {'Test':>6} {'RMSE':>9} {'MAE':>9} {'MAPE%':>8} {'R²':>8}")
    print("   " + "-" * 55)
    for r in cv_results:
        r2 = f"{r['R2']:.4f}" if r.get('R2') is not None else "N/A"
        print(f"   {r['fold']:>5} {r['train_size']:>7} {r['test_size']:>6} "
              f"Rs {r['RMSE']:>6.2f}  Rs {r['MAE']:>6.2f} "
              f"  {r['MAPE']:>6.1f}%  {r2:>8}")
    avg_rmse = np.mean([r['RMSE'] for r in cv_results])
    avg_mape = np.mean([r['MAPE'] for r in cv_results])
    print(f"\n   Average RMSE: Rs {avg_rmse:.2f} | Average MAPE: {avg_mape:.1f}%")
else:
    print("   ℹ️  Not enough data for cross-validation")

# ──────────────────────────────────────────────
# PHASE C — MODEL COMPARISON
# ──────────────────────────────────────────────
print("\n" + "-" * 65)
print("🔬 PHASE C — Model Comparison (Prophet vs MA vs LR)...")

comparison, all_forecasts = compare_all_models(daily_df, holidays, FORECAST_DAYS)

if comparison:
    print(f"\n   {'Model':<22} {'RMSE':>9} {'MAE':>9} {'MAPE%':>8} {'R²':>8}")
    print("   " + "-" * 60)
    for model_name, mets in comparison.items():
        r2 = f"{mets['R2']:.4f}" if mets.get('R2') is not None else "N/A"
        print(f"   {model_name:<22} Rs {mets['RMSE']:>6.2f}  "
              f"Rs {mets['MAE']:>6.2f}  {mets['MAPE']:>6.1f}%  {r2:>8}")

    best_overall = min(comparison, key=lambda k: comparison[k]['RMSE']
                       if comparison[k]['RMSE'] else float('inf'))
    print(f"\n   🏆 Best overall model: {best_overall}")

# ──────────────────────────────────────────────
# PHASE D — FINAL PROPHET MODEL
# ──────────────────────────────────────────────
print("\n" + "-" * 65)
print("🔮 PHASE D — Final Model + 30-day Forecast...")

_, final_forecast = run_prophet(daily_df, holidays, FORECAST_DAYS, best_cp, best_sp, best_mode)
fut_only = final_forecast[final_forecast['ds'] > daily_df['ds'].max()]
avg_spend = daily_df['y'].mean()

print(f"\n   Forecast Next {FORECAST_DAYS} Days:")
print(f"   {'Date':<12} {'Forecast Rs':>13} {'Lower Rs':>10} {'Upper Rs':>10}  Spike?")
print("   " + "-" * 55)
for _, row in fut_only.iterrows():
    spike = " ⚠️ SPIKE" if row['yhat'] > avg_spend * SPIKE_MULT else ""
    print(f"   {str(row['ds'].date()):<12} Rs {row['yhat']:>9.2f} "
          f"  Rs {row['yhat_lower']:>6.2f}  Rs {row['yhat_upper']:>6.2f}{spike}")

print(f"\n   Total 30-day forecast: Rs {fut_only['yhat'].sum():,.2f}")
print(f"   Avg daily forecast   : Rs {fut_only['yhat'].mean():,.2f}")

# ──────────────────────────────────────────────
# PHASE E — ANOMALY DETECTION
# ──────────────────────────────────────────────
print("\n" + "-" * 65)
print(f"🚨 PHASE E — Anomaly Detection ({ANOMALY_SIGMA}σ threshold)...")

anom_df    = detect_anomalies(daily_df, final_forecast, ANOMALY_SIGMA)
anomalies  = anom_df[anom_df['anomaly']]

print(f"   Total days analyzed : {len(anom_df)}")
print(f"   Anomalies detected  : {len(anomalies)}")

if len(anomalies) > 0:
    print(f"\n   {'Date':<12} {'Actual Rs':>10} {'Predicted Rs':>13} {'Residual':>10} {'Severity':>10} Direction")
    print("   " + "-" * 65)
    for _, row in anomalies.iterrows():
        direction = "HIGH ↑" if row['residual'] < 0 else "LOW ↓"
        print(f"   {str(row['ds'].date()):<12} Rs {row['y']:>7.0f}  "
              f"Rs {row['yhat']:>9.0f}  Rs {row['residual']:>7.0f}  "
              f"{row['severity']:>8.1f}σ  {direction}")

# ──────────────────────────────────────────────
# PHASE F — WEEKLY + MONTHLY SUMMARY
# ──────────────────────────────────────────────
print("\n" + "-" * 65)
print("📅 PHASE F — Weekly & Monthly Forecast Summary...")

fut_only = fut_only.copy()
fut_only['week']  = fut_only['ds'].dt.to_period('W')
fut_only['month'] = fut_only['ds'].dt.to_period('M')

weekly  = fut_only.groupby('week') [['yhat','yhat_lower','yhat_upper']].sum()
monthly = fut_only.groupby('month')[['yhat','yhat_lower','yhat_upper']].sum()

print("\n   Weekly Forecast:")
print(f"   {'Week':<22} {'Forecast Rs':>13} {'Lower Rs':>11} {'Upper Rs':>11}")
print("   " + "-" * 60)
for week, row in weekly.iterrows():
    print(f"   {str(week):<22} Rs {row['yhat']:>9.2f}  "
          f"Rs {row['yhat_lower']:>7.2f}  Rs {row['yhat_upper']:>7.2f}")

print("\n   Monthly Forecast:")
print(f"   {'Month':<12} {'Forecast Rs':>13} {'Lower Rs':>11} {'Upper Rs':>11}")
print("   " + "-" * 50)
for month, row in monthly.iterrows():
    print(f"   {str(month):<12} Rs {row['yhat']:>9.2f}  "
          f"Rs {row['yhat_lower']:>7.2f}  Rs {row['yhat_upper']:>7.2f}")

# ──────────────────────────────────────────────
# SAVE 8-SHEET EXCEL
# ──────────────────────────────────────────────
print("\n" + "-" * 65)
print("💾 Saving 8-sheet Excel report → forecast_output_advanced.xlsx...")

with pd.ExcelWriter("forecast_output_advanced.xlsx", engine='openpyxl') as writer:

    # Sheet 1: Full Forecast
    final_forecast[['ds','yhat','yhat_lower','yhat_upper','trend']].rename(
        columns={'ds':'Date','yhat':'Forecast','yhat_lower':'Lower',
                 'yhat_upper':'Upper','trend':'Trend'}
    ).to_excel(writer, sheet_name='Full Forecast', index=False)

    # Sheet 2: Future Only
    fut_only[['ds','yhat','yhat_lower','yhat_upper']].rename(
        columns={'ds':'Date','yhat':'Forecast Rs',
                 'yhat_lower':'Lower Rs','yhat_upper':'Upper Rs'}
    ).to_excel(writer, sheet_name='Future Predictions', index=False)

    # Sheet 3: CV Results
    if cv_results:
        pd.DataFrame(cv_results).to_excel(
            writer, sheet_name='Cross Validation', index=False)

    # Sheet 4: Anomalies
    if len(anomalies) > 0:
        anomalies[['ds','y','yhat','residual','severity']].rename(
            columns={'ds':'Date','y':'Actual Rs','yhat':'Predicted Rs'}
        ).to_excel(writer, sheet_name='Anomalies', index=False)
    else:
        pd.DataFrame(columns=['Date','Actual Rs','Predicted Rs',
                               'residual','severity']).to_excel(
            writer, sheet_name='Anomalies', index=False)

    # Sheet 5: Weekly Summary
    weekly.reset_index().rename(
        columns={'week':'Week','yhat':'Forecast Rs',
                 'yhat_lower':'Lower Rs','yhat_upper':'Upper Rs'}
    ).to_excel(writer, sheet_name='Weekly Summary', index=False)

    # Sheet 6: Monthly Summary
    monthly.reset_index().rename(
        columns={'month':'Month','yhat':'Forecast Rs',
                 'yhat_lower':'Lower Rs','yhat_upper':'Upper Rs'}
    ).to_excel(writer, sheet_name='Monthly Summary', index=False)

    # Sheet 7: Hyperparameter Tuning
    if not tune_df.empty:
        tune_df.to_excel(writer, sheet_name='Hyperparameter Tuning', index=False)

    # Sheet 8: Model Comparison
    if comparison:
        comp_rows = [{'Model': k, **v} for k, v in comparison.items()]
        pd.DataFrame(comp_rows).to_excel(
            writer, sheet_name='Model Comparison', index=False)

print("   ✅ Saved → forecast_output_advanced.xlsx")

# ──────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ✅ ADVANCED FORECASTING COMPLETE")
print("=" * 65)
print(f"  Total {FORECAST_DAYS}-day Forecast: Rs {fut_only['yhat'].sum():,.2f}")
print(f"  Anomalies Found          : {len(anomalies)}")
print(f"  Best CV RMSE             : Rs {min(r['RMSE'] for r in cv_results):.2f}"
      if cv_results else "  CV not run")
print()
print("  📁 Output Files:")
print("     ✅ forecast_output_advanced.xlsx  (8 sheets)")
print()
print("  ▶️  Next → streamlit run advanced_app.py")
print("=" * 65)
