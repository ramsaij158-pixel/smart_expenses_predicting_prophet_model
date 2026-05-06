"""
models.py
=========
Advanced Model Training & Comparison
Supports: Prophet, Linear Regression, Moving Average
Includes: Hyperparameter tuning, Cross-validation, Model comparison
"""

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import itertools
import warnings
warnings.filterwarnings('ignore')

CATEGORIES = ['Food', 'Transport', 'Entertainment', 'Travel', 'Medical',
              'Education', 'Shopping', 'Other']

# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return dict(MSE=None, RMSE=None, MAE=None, R2=None, MAPE=None)
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    try:
        r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else None
    except Exception:
        r2 = None
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    return dict(
        MSE =round(mse, 3),
        RMSE=round(rmse, 3),
        MAE =round(mae, 3),
        R2  =round(r2, 4) if r2 is not None else None,
        MAPE=round(mape, 2)
    )

# ──────────────────────────────────────────────
# PROPHET MODEL
# ──────────────────────────────────────────────
def build_prophet(holidays_df, cp=0.05, sp=1.0, mode='additive'):
    h = holidays_df if (holidays_df is not None and len(holidays_df) > 0) else None
    return Prophet(
        holidays=h,
        changepoint_prior_scale=cp,
        seasonality_prior_scale=sp,
        seasonality_mode=mode,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

def run_prophet(daily_df, holidays_df, forecast_days=30, cp=0.05, sp=1.0, mode='additive'):
    model = build_prophet(holidays_df, cp, sp, mode)
    model.fit(daily_df)
    future   = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    return model, forecast

def hyperparameter_tuning(daily_df, holidays_df, test_size=2):
    """Grid search over 24 parameter combinations"""
    if len(daily_df) < 5:
        return pd.DataFrame(), 0.05, 1.0, 'additive'

    param_grid = {
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.3],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'seasonality_mode'       : ['additive', 'multiplicative'],
    }
    combos = list(itertools.product(*param_grid.values()))
    keys   = list(param_grid.keys())

    ts    = min(test_size, len(daily_df) - 3)
    train = daily_df.iloc[:-ts].copy()
    test  = daily_df.iloc[-ts:].copy()

    best_rmse = float('inf')
    best_cp, best_sp, best_mode = 0.05, 1.0, 'additive'
    rows = []

    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            m  = build_prophet(holidays_df, params['changepoint_prior_scale'],
                               params['seasonality_prior_scale'], params['seasonality_mode'])
            m.fit(train)
            fc    = m.predict(m.make_future_dataframe(periods=len(test)))
            preds = fc[fc['ds'].isin(test['ds'])]['yhat'].values
            acts  = test['y'].values[:len(preds)]
            if len(acts) == 0:
                continue
            met = compute_metrics(acts, preds)
            met.update(**params)
            rows.append(met)
            if met['RMSE'] and met['RMSE'] < best_rmse:
                best_rmse = met['RMSE']
                best_cp   = params['changepoint_prior_scale']
                best_sp   = params['seasonality_prior_scale']
                best_mode = params['seasonality_mode']
        except Exception:
            continue

    tune_df = pd.DataFrame(rows)
    if not tune_df.empty and 'RMSE' in tune_df.columns:
        tune_df = tune_df.sort_values('RMSE').reset_index(drop=True)

    return tune_df, best_cp, best_sp, best_mode

def cross_validate(daily_df, holidays_df, cp, sp, mode, n_folds=3, test_size=1):
    """Walk-forward cross-validation"""
    results = []
    n = len(daily_df)
    for fold in range(n_folds):
        end = n - test_size * (n_folds - fold)
        if end < 5:
            continue
        train = daily_df.iloc[:end].copy()
        test  = daily_df.iloc[end:end + test_size].copy()
        if len(test) == 0:
            continue
        try:
            m  = build_prophet(holidays_df, cp, sp, mode)
            m.fit(train)
            fc    = m.predict(m.make_future_dataframe(periods=len(test)))
            preds = fc[fc['ds'].isin(test['ds'])]['yhat'].values
            acts  = test['y'].values[:len(preds)]
            if len(acts) == 0:
                continue
            met = compute_metrics(acts, preds)
            met['fold']       = fold + 1
            met['train_size'] = len(train)
            met['test_size']  = len(acts)
            results.append(met)
        except Exception:
            continue
    return results

# ──────────────────────────────────────────────
# MOVING AVERAGE MODEL
# ──────────────────────────────────────────────
def run_moving_average(daily_df, forecast_days=30, window=7):
    series = daily_df.set_index('ds')['y']
    ma_vals = series.rolling(window=window, min_periods=1).mean()
    last_ma = float(ma_vals.iloc[-1])

    last_date    = daily_df['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    future_df = pd.DataFrame({
        'ds'        : future_dates,
        'yhat'      : last_ma,
        'yhat_lower': last_ma * 0.85,
        'yhat_upper': last_ma * 1.15,
    })
    in_sample = pd.DataFrame({'ds': daily_df['ds'], 'yhat': ma_vals.values})
    return future_df, in_sample

# ──────────────────────────────────────────────
# LINEAR REGRESSION MODEL
# ──────────────────────────────────────────────
def run_linear_regression(daily_df, forecast_days=30):
    df = daily_df.copy().reset_index(drop=True)
    df['t'] = np.arange(len(df))

    model = LinearRegression()
    model.fit(df[['t']], df['y'])

    future_t     = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    future_dates = pd.date_range(
        start=df['ds'].max() + pd.Timedelta(days=1), periods=forecast_days)
    future_preds = model.predict(future_t)

    future_df = pd.DataFrame({
        'ds'        : future_dates,
        'yhat'      : future_preds,
        'yhat_lower': future_preds * 0.85,
        'yhat_upper': future_preds * 1.15,
    })
    in_sample = pd.DataFrame({'ds': df['ds'], 'yhat': model.predict(df[['t']])})
    return future_df, in_sample

# ──────────────────────────────────────────────
# MODEL COMPARISON
# ──────────────────────────────────────────────
def compare_all_models(daily_df, holidays_df, forecast_days=30):
    """Train Prophet, Moving Average, Linear Regression and compare"""
    if len(daily_df) < 4:
        return {}, {}

    test_size = max(1, len(daily_df) // 5)
    train = daily_df.iloc[:-test_size].copy()
    test  = daily_df.iloc[-test_size:].copy()
    acts  = test['y'].values

    comparison  = {}
    forecasts   = {}

    # 1. Prophet
    try:
        _, fc_p = run_prophet(train, holidays_df, test_size)
        preds_p = fc_p[fc_p['ds'].isin(test['ds'])]['yhat'].values
        if len(preds_p) > 0:
            comparison['Prophet'] = compute_metrics(acts[:len(preds_p)], preds_p)
        _, fc_full = run_prophet(daily_df, holidays_df, forecast_days)
        forecasts['Prophet'] = fc_full[fc_full['ds'] > daily_df['ds'].max()]
    except Exception:
        pass

    # 2. Moving Average
    try:
        fut_ma, in_ma = run_moving_average(train, test_size)
        preds_ma = np.full(len(test), in_ma['yhat'].iloc[-1])
        comparison['Moving Average (7d)'] = compute_metrics(acts, preds_ma)
        fut_ma_full, _ = run_moving_average(daily_df, forecast_days)
        forecasts['Moving Average'] = fut_ma_full
    except Exception:
        pass

    # 3. Linear Regression
    try:
        fut_lr, in_lr = run_linear_regression(train, test_size)
        preds_lr = fut_lr['yhat'].values[:len(test)]
        comparison['Linear Regression'] = compute_metrics(acts[:len(preds_lr)], preds_lr)
        fut_lr_full, _ = run_linear_regression(daily_df, forecast_days)
        forecasts['Linear Regression'] = fut_lr_full
    except Exception:
        pass

    return comparison, forecasts

# ──────────────────────────────────────────────
# ANOMALY DETECTION
# ──────────────────────────────────────────────
def detect_anomalies(daily_df, forecast_df, sigma=2.0):
    merged = daily_df.merge(
        forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')
    merged['residual'] = merged['y'] - merged['yhat']
    std = merged['residual'].std()
    if std == 0:
        merged['anomaly']  = False
        merged['severity'] = 0.0
    else:
        merged['anomaly']  = np.abs(merged['residual']) > sigma * std
        merged['severity'] = (np.abs(merged['residual']) / (std + 1e-9)).round(2)
    return merged

# ──────────────────────────────────────────────
# AUTO CATEGORY CLASSIFICATION
# ──────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    'Food'         : ['zomato', 'swiggy', 'restaurant', 'food', 'lunch', 'dinner',
                      'breakfast', 'canteen', 'cafe', 'coffee', 'tea', 'snacks',
                      'pizza', 'burger', 'biryani', 'mess', 'tiffin'],
    'Transport'    : ['ola', 'uber', 'auto', 'bus', 'train', 'metro', 'taxi',
                      'petrol', 'fuel', 'cab', 'rapido', 'rickshaw', 'railway'],
    'Entertainment': ['movie', 'netflix', 'amazon prime', 'spotify', 'game', 'concert',
                      'theatre', 'club', 'party', 'event', 'fest', 'pvr', 'inox'],
    'Travel'       : ['flight', 'hotel', 'trip', 'travel', 'holiday', 'vacation',
                      'booking', 'airbnb', 'hostel', 'tour', 'makemytrip'],
    'Medical'      : ['medicine', 'doctor', 'hospital', 'pharmacy', 'clinic',
                      'health', 'apollo', 'medplus', 'consultation'],
    'Education'    : ['book', 'course', 'udemy', 'coursera', 'fees', 'stationery',
                      'pen', 'notebook', 'library', 'tuition', 'coaching'],
    'Shopping'     : ['amazon', 'flipkart', 'myntra', 'clothes', 'shirt', 'shoes',
                      'mall', 'grocery', 'supermarket', 'big bazaar', 'dmart'],
}

def classify_category(description: str) -> str:
    desc = description.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(k in desc for k in keywords):
            return category
    return 'Other'
