"""
advanced_app.py
===============
Advanced Student Expense Forecaster — Full Dashboard
Run with: streamlit run advanced_app.py

Features:
  ✅ SQLite database (no more Excel files)
  ✅ Real-time expense entry with auto-categorization
  ✅ Budget tracking with progress bars
  ✅ Prophet + Linear Regression + Moving Average comparison
  ✅ Hyperparameter tuning (24 combinations)
  ✅ Cross-validation (walk-forward)
  ✅ Anomaly detection with AI explanations
  ✅ Claude AI personalized spending advice
  ✅ Weekly & Monthly reports
  ✅ Edit / Delete expenses
  ✅ Export full report to Excel
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings("ignore")

# ── Local modules ──
from database   import (init_db, seed_from_excel, add_expense, get_all_expenses,
                         get_daily_totals, get_category_grouped, get_holidays,
                         delete_expense, update_expense, set_budget, get_budgets,
                         get_current_month_spending, get_summary_stats)
from models     import (run_prophet, hyperparameter_tuning, cross_validate,
                         compare_all_models, detect_anomalies, classify_category,
                         CATEGORIES, compute_metrics)
from ai_advisor import (get_spending_advice, explain_anomaly,
                         get_budget_recommendation, get_weekly_summary)

# ══════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Expense Forecaster",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════
# CSS — Dark refined theme
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #09090b; }

.kpi-card {
  background: linear-gradient(135deg, #18181b, #27272a);
  border: 1px solid #3f3f46; border-radius: 14px;
  padding: 20px 22px; text-align: center;
  transition: border-color .2s;
}
.kpi-card:hover { border-color: #71717a; }
.kpi-label { color: #71717a; font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.kpi-value { color: #fafafa; font-size: 28px; font-weight: 600; }
.kpi-sub   { color: #52525b; font-size: 12px; margin-top: 4px; }

.section-title {
  color: #e4e4e7; font-size: 16px; font-weight: 600;
  margin: 24px 0 12px; padding-bottom: 8px;
  border-bottom: 1px solid #27272a;
}

.budget-bar-container { margin: 8px 0; }
.budget-label { color: #a1a1aa; font-size: 13px; margin-bottom: 4px; }
.budget-bar-bg { background: #27272a; border-radius: 6px; height: 10px; overflow: hidden; }
.budget-bar-fill { height: 100%; border-radius: 6px; transition: width .3s; }
.budget-info { color: #71717a; font-size: 12px; margin-top: 3px; }

.badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
  font-size: 11px; font-weight: 600; }
.badge-green  { background: #052e16; color: #4ade80; }
.badge-yellow { background: #422006; color: #fbbf24; }
.badge-red    { background: #450a0a; color: #f87171; }
.badge-blue   { background: #082f49; color: #38bdf8; }

.ai-box { background: #0c1a0c; border: 1px solid #166534;
  border-radius: 12px; padding: 18px 20px;
  color: #bbf7d0; font-size: 14px; line-height: 1.7; }
.ai-box strong { color: #4ade80; }

.anomaly-card { background: #1c0a0a; border: 1px solid #7f1d1d;
  border-radius: 10px; padding: 14px 16px; margin: 8px 0; }
.anomaly-date { color: #fca5a5; font-size: 13px; font-weight: 600; }
.anomaly-detail { color: #ef4444; font-size: 12px; margin-top: 4px; }
.anomaly-explain { color: #fca5a5; font-size: 13px; margin-top: 8px; line-height: 1.5; }

div[data-testid="stSidebar"] { background: #09090b; border-right: 1px solid #27272a; }
.stTabs [data-baseweb="tab"] { 
  background: #18181b; color: #71717a;
  border-radius: 8px 8px 0 0; padding: 10px 18px; font-weight: 500; font-size: 13px;
}
.stTabs [aria-selected="true"] { background: #2563eb !important; color: #fff !important; }

.stDataFrame { border: 1px solid #27272a; border-radius: 10px; }
.stForm { background: #18181b; border: 1px solid #27272a; border-radius: 14px; padding: 20px; }
</style>
""", unsafe_allow_html=True)

COLORS = ['#3b82f6','#22c55e','#f97316','#ec4899','#8b5cf6',
          '#06b6d4','#eab308','#ef4444','#14b8a6','#f59e0b']
def layout(height=350, title=None, margin_t=36, legend=None, xaxis=None, yaxis=None):
    """Clean helper — builds plotly layout dict without key conflicts."""
    d = dict(
        template      = 'plotly_dark',
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor  = 'rgba(0,0,0,0)',
        margin        = dict(l=0, r=0, t=margin_t, b=0),
        font          = dict(family='DM Sans'),
        height        = height,
    )
    if title  : d['title']  = title
    if legend : d['legend'] = legend
    if xaxis  : d['xaxis']  = xaxis
    if yaxis  : d['yaxis']  = yaxis
    return d

# keep alias so any remaining **DARK_LAYOUT refs still work
DARK_LAYOUT = dict(
    template      = 'plotly_dark',
    paper_bgcolor = 'rgba(0,0,0,0)',
    plot_bgcolor  = 'rgba(0,0,0,0)',
    font          = dict(family='DM Sans'),
)

# ══════════════════════════════════════════════════════
# INIT DATABASE
# ══════════════════════════════════════════════════════
init_db()
seeded, msg = seed_from_excel()

# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 💡 Smart Expense Forecaster")
    st.markdown("---")

    st.markdown("### ⚙️ Forecast Settings")
    forecast_days = st.slider("Forecast Days",       14, 90, 30)
    cp_scale      = st.slider("Changepoint Prior",   0.01, 0.5, 0.05, 0.01)
    sp_scale      = st.slider("Seasonality Prior",   0.1,  10.0, 1.0, 0.1)
    seas_mode     = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])
    anomaly_sigma = st.slider("Anomaly Threshold σ", 1.0, 3.0, 2.0, 0.1)

    st.markdown("---")
    st.markdown("### 🔔 Alert Settings")
    spike_threshold = st.number_input("Spike Alert (Rs/day)", value=300, step=50)

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    stats = get_summary_stats()
    if stats:
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Spend", "Rs {:,.0f}".format(stats["total_spend"]))
        st.metric("Records",        stats['total_records'])
        st.metric("Top Category",   stats['top_category'])

# ══════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def load_data():
    expenses   = get_all_expenses()
    daily_df   = get_daily_totals()
    cat_df     = get_category_grouped()
    holidays   = get_holidays()
    budgets    = get_budgets()
    return expenses, daily_df, cat_df, holidays, budgets

expenses_df, daily_df, cat_df, holidays_df, budgets_df = load_data()
has_data = not daily_df.empty and len(daily_df) >= 3

# ══════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Dashboard",
    "➕ Add Expense",
    "💰 Budget Tracker",
    "🔮 Forecast",
    "📈 Model Comparison",
    "🚨 Anomalies",
    "🤖 AI Advisor",
    "📅 Reports",
    "📤 Export",
])
(tab_dash, tab_add, tab_budget, tab_forecast,
 tab_compare, tab_anomaly, tab_ai, tab_report, tab_export) = tabs

# ══════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════
with tab_dash:
    if not has_data:
        st.info("No data yet. Add expenses in the ➕ Add Expense tab.")
    else:
        stats = get_summary_stats()

        # KPI row
        c1, c2, c3, c4, c5 = st.columns(5)
        kpis = [
            (c1, "Total Spend",    "Rs {:,.0f}".format(stats["total_spend"]), "all time"),
            (c2, "Avg Daily",      "Rs {:,.0f}".format(stats["avg_daily"]), "per day"),
            (c3, "Peak Day",       "Rs {:,.0f}".format(stats["max_day"]), "highest"),
            (c4, "Categories",     str(stats['categories']),               "tracked"),
            (c5, "Total Records",  str(stats['total_records']),            "entries"),
        ]
        for col, label, val, sub in kpis:
            with col:
                kpi_html = (
                    '<div class="kpi-card">'
                    f'<div class="kpi-label">{label}</div>'
                    f'<div class="kpi-value">{val}</div>'
                    f'<div class="kpi-sub">{sub}</div>'
                    '</div>'
                )
                st.markdown(kpi_html, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Daily Spending History</div>',
                    unsafe_allow_html=True)
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Bar(
            x=daily_df['ds'], y=daily_df['y'],
            marker_color='#3b82f6', opacity=0.8, name='Daily Spend'))
        ma7 = daily_df['y'].rolling(7, min_periods=1).mean()
        fig_daily.add_trace(go.Scatter(
            x=daily_df['ds'], y=ma7,
            line=dict(color='#f97316', width=2), name='7-day MA'))
        fig_daily.update_layout(**layout(height=300,
            xaxis=dict(gridcolor='#27272a'), yaxis=dict(gridcolor='#27272a', title='Rs')))
        st.plotly_chart(fig_daily, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-title">Spend by Category</div>',
                        unsafe_allow_html=True)
            cat_sum = cat_df.groupby('Category')['Amount'].sum().reset_index()
            fig_pie = px.pie(cat_sum, values='Amount', names='Category',
                             color_discrete_sequence=COLORS)
            fig_pie.update_layout(**layout(height=280, margin_t=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.markdown('<div class="section-title">Category Trend Over Time</div>',
                        unsafe_allow_html=True)
            fig_cat = go.Figure()
            for i, cat in enumerate(cat_df['Category'].unique()):
                sub = cat_df[cat_df['Category'] == cat]
                fig_cat.add_trace(go.Scatter(
                    x=sub['Date'], y=sub['Amount'], mode='lines+markers',
                    name=cat, line=dict(color=COLORS[i % len(COLORS)], width=2)))
            fig_cat.update_layout(**layout(height=280,
                xaxis=dict(gridcolor='#27272a'), yaxis=dict(gridcolor='#27272a')))
            st.plotly_chart(fig_cat, use_container_width=True)

        # Recent transactions
        st.markdown('<div class="section-title">Recent Transactions</div>',
                    unsafe_allow_html=True)
        recent = expenses_df.head(10)[['date','category','amount','description','payment_mode']]
        st.dataframe(recent.rename(columns={
            'date': 'Date', 'category': 'Category', 'amount': 'Amount (Rs)',
            'description': 'Description', 'payment_mode': 'Payment'}),
            use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
# TAB 2 — ADD EXPENSE
# ══════════════════════════════════════════════════════
with tab_add:
    st.markdown('<div class="section-title">Add New Expense</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        with st.form("add_expense_form", clear_on_submit=True):
            col_a, col_b = st.columns(2)
            with col_a:
                exp_date   = st.date_input("Date", value=datetime.today())
                exp_amount = st.number_input("Amount (Rs)", min_value=0.01, step=1.0)
            with col_b:
                exp_desc = st.text_input("Description (optional)",
                                          placeholder="e.g. Zomato order, Ola cab")
                if exp_desc:
                    suggested = classify_category(exp_desc)
                    st.caption(f"💡 Suggested category: **{suggested}**")
                else:
                    suggested = "Food"

                exp_cat = st.selectbox("Category", CATEGORIES,
                                        index=CATEGORIES.index(suggested)
                                        if suggested in CATEGORIES else 0)

            col_c, col_d = st.columns(2)
            with col_c:
                exp_payment = st.selectbox("Payment Mode",
                    ["Cash", "UPI", "Card", "Net Banking", "Other"])
            with col_d:
                exp_event = st.text_input("Event/Holiday (optional)",
                                           placeholder="e.g. Diwali, College Fest")

            submitted = st.form_submit_button("➕ Add Expense", use_container_width=True,
                                               type="primary")
            if submitted:
                if exp_amount <= 0:
                    st.error("Amount must be greater than 0")
                else:
                    add_expense(exp_date, exp_amount, exp_cat,
                                exp_desc, exp_payment, exp_event)
                    st.cache_data.clear()
                    st.success(f"✅ Added Rs {exp_amount:.0f} for {exp_cat} on {exp_date}")
                    st.rerun()

    with c2:
        st.markdown('<div class="section-title">Auto-Category Guide</div>',
                    unsafe_allow_html=True)
        guide = {
            '🍕 Food'          : 'zomato, swiggy, restaurant, canteen, cafe',
            '🚌 Transport'     : 'ola, uber, bus, train, metro, rapido',
            '🎬 Entertainment' : 'movie, netflix, pvr, concert, fest',
            '✈️ Travel'        : 'flight, hotel, trip, booking, holiday',
            '💊 Medical'       : 'medicine, doctor, pharmacy, hospital',
            '📚 Education'     : 'books, course, udemy, coaching, fees',
            '🛍️ Shopping'      : 'amazon, flipkart, clothes, grocery',
        }
        for cat, keywords in guide.items():
            st.caption(f"**{cat}**: {keywords}")

    # Edit / Delete existing
    st.markdown('<div class="section-title">Manage Existing Expenses</div>',
                unsafe_allow_html=True)
    if not expenses_df.empty:
        show_df = expenses_df[['id','date','category','amount','description','payment_mode']].copy()
        show_df['date'] = show_df['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(show_df.rename(columns={
            'id': 'ID', 'date': 'Date', 'category': 'Category',
            'amount': 'Amount', 'description': 'Description', 'payment_mode': 'Payment'}),
            use_container_width=True, hide_index=True)

        del_id = st.number_input("Enter ID to delete", min_value=1, step=1)
        if st.button("🗑️ Delete Expense", type="secondary"):
            delete_expense(int(del_id))
            st.cache_data.clear()
            st.success(f"Deleted expense #{del_id}")
            st.rerun()
    else:
        st.info("No expenses yet. Add one above.")

# ══════════════════════════════════════════════════════
# TAB 3 — BUDGET TRACKER
# ══════════════════════════════════════════════════════
with tab_budget:
    st.markdown('<div class="section-title">Set Monthly Budgets</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        with st.form("budget_form"):
            st.markdown("**Set budget for each category (Rs/month)**")
            budget_inputs = {}
            existing = dict(zip(budgets_df['category'], budgets_df['monthly_limit'])) \
                       if not budgets_df.empty else {}

            for cat in CATEGORIES:
                default_val = float(existing.get(cat, 0))
                budget_inputs[cat] = st.number_input(
                    cat, min_value=0.0, value=default_val, step=100.0, key=f"bud_{cat}")

            if st.form_submit_button("💾 Save Budgets", use_container_width=True,
                                      type="primary"):
                for cat, limit in budget_inputs.items():
                    if limit > 0:
                        set_budget(cat, limit)
                st.cache_data.clear()
                st.success("✅ Budgets saved!")
                st.rerun()

    with c2:
        st.markdown('<div class="section-title">This Month — Budget vs Actual</div>',
                    unsafe_allow_html=True)
        month_spend  = get_current_month_spending()
        budgets_now  = get_budgets()

        if budgets_now.empty:
            st.info("Set budgets on the left first.")
        else:
            for _, brow in budgets_now.iterrows():
                cat   = brow['category']
                limit = brow['monthly_limit']
                spent_row = month_spend[month_spend['category'] == cat]
                spent = float(spent_row['spent'].iloc[0]) if not spent_row.empty else 0
                pct   = (spent / limit * 100) if limit > 0 else 0

                if pct >= 100:
                    color, badge = '#ef4444', f'<span class="badge badge-red">OVER</span>'
                elif pct >= 80:
                    color, badge = '#f97316', f'<span class="badge badge-yellow">WARNING</span>'
                else:
                    color, badge = '#22c55e', f'<span class="badge badge-green">OK</span>'

                pct_display = min(pct, 100)
                budget_html = (
                    '<div class="budget-bar-container">'
                    f'<div class="budget-label">{cat} {badge}</div>'
                    '<div class="budget-bar-bg">'
                    f'<div class="budget-bar-fill" style="width:{pct_display:.1f}%; background:{color};"></div>'
                    '</div>'
                    f'<div class="budget-info">Rs {spent:.0f} / Rs {limit:.0f} &nbsp;|&nbsp; {pct:.1f}% used</div>'
                    '</div>'
                )
                st.markdown(budget_html, unsafe_allow_html=True)

    # Budget AI recommendation
    st.markdown('<div class="section-title">🤖 AI Budget Recommendation</div>',
                unsafe_allow_html=True)
    if st.button("Get AI Budget Suggestions", type="primary"):
        if has_data:
            cat_totals = dict(cat_df.groupby('Category')['Amount'].sum())
            date_range = (daily_df['ds'].max() - daily_df['ds'].min()).days / 30
            months     = max(date_range, 0.5)
            with st.spinner("Getting AI recommendations..."):
                advice = get_budget_recommendation(cat_totals, months)
            st.markdown(f'<div class="ai-box">{advice}</div>', unsafe_allow_html=True)
        else:
            st.warning("Need at least 3 days of data for recommendations.")

# ══════════════════════════════════════════════════════
# TAB 4 — FORECAST
# ══════════════════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="section-title">Prophet Forecast</div>', unsafe_allow_html=True)

    if not has_data:
        st.info("Not enough data for forecasting. Add at least 3 expenses.")
    else:
        run_tune = st.checkbox("🎯 Run Hyperparameter Tuning (24 combos)", value=False)

        if st.button("🔮 Run Forecast", type="primary", use_container_width=True):
            with st.spinner("Training Prophet model..."):
                if run_tune:
                    st.info("Running grid search over 24 combinations...")
                    tune_df, best_cp, best_sp, best_mode = hyperparameter_tuning(
                        daily_df, holidays_df)
                    st.session_state['best_params'] = {
                        'cp': best_cp, 'sp': best_sp, 'mode': best_mode}
                    st.success(f"✅ Best: CP={best_cp} SP={best_sp} Mode={best_mode}")
                    if not tune_df.empty:
                        st.dataframe(tune_df.round(3), use_container_width=True)
                else:
                    best_cp, best_sp, best_mode = cp_scale, sp_scale, seas_mode

                # Cross-validation
                cv_results = cross_validate(
                    daily_df, holidays_df, best_cp, best_sp, best_mode)

                # Final model
                _, forecast = run_prophet(
                    daily_df, holidays_df, forecast_days, best_cp, best_sp, best_mode)

                st.session_state['forecast']   = forecast
                st.session_state['cv_results'] = cv_results

        if 'forecast' in st.session_state:
            forecast = st.session_state['forecast']
            cv_res   = st.session_state.get('cv_results', [])
            fut_only = forecast[forecast['ds'] > daily_df['ds'].max()]

            # Main forecast chart
            fig = go.Figure()
            fig.add_trace(go.Bar(x=daily_df['ds'], y=daily_df['y'],
                name='Actual', marker_color='#3b82f6', opacity=0.7))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                name='Forecast', line=dict(color='#22c55e', width=2.5)))
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast['ds'], forecast['ds'][::-1]]),
                y=pd.concat([forecast['yhat_upper'], forecast['yhat_lower'][::-1]]),
                fill='toself', fillcolor='rgba(34,197,94,0.1)',
                line=dict(color='rgba(0,0,0,0)'), name='Confidence Band'))
            vline_x = daily_df['ds'].max().timestamp() * 1000
            fig.add_vline(x=vline_x, line_dash='dash',
                          line_color='#71717a', annotation_text='Forecast starts')
            fig.update_layout(**layout(height=360,
                xaxis=dict(gridcolor='#27272a'), yaxis=dict(gridcolor='#27272a', title='Rs'),
                legend=dict(orientation='h', y=1.08)))
            st.plotly_chart(fig, use_container_width=True)

            # Metrics row
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("30-day Forecast Total", "Rs {:,.0f}".format(fut_only["yhat"].sum()))
            with c2:
                st.metric("Daily Average (forecast)", "Rs {:,.0f}".format(fut_only["yhat"].mean()))
            with c3:
                spikes = fut_only[fut_only['yhat'] > spike_threshold]
                st.metric("Spike Days", f"{len(spikes)} days > Rs {spike_threshold}")

            # CV results
            if cv_res:
                st.markdown('<div class="section-title">Cross-Validation Results</div>',
                            unsafe_allow_html=True)
                cv_df = pd.DataFrame(cv_res)
                st.dataframe(cv_df.round(3), use_container_width=True, hide_index=True)

                avg_rmse = np.mean([r['RMSE'] for r in cv_res])
                avg_mape = np.mean([r['MAPE'] for r in cv_res])
                st.caption(f"Average RMSE: Rs {avg_rmse:.2f} | Average MAPE: {avg_mape:.1f}%")

            # Future predictions table
            st.markdown('<div class="section-title">Day-by-Day Forecast</div>',
                        unsafe_allow_html=True)
            fut_display = fut_only[['ds','yhat','yhat_lower','yhat_upper']].copy()
            fut_display.columns = ['Date','Forecast Rs','Lower Rs','Upper Rs']
            fut_display['Date'] = fut_display['Date'].dt.strftime('%Y-%m-%d')
            fut_display['Spike?'] = fut_display['Forecast Rs'].apply(
                lambda x: '⚠️ YES' if x > spike_threshold else '✅ No')
            st.dataframe(fut_display.round(2), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
# TAB 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════
with tab_compare:
    st.markdown('<div class="section-title">Prophet vs Linear Regression vs Moving Average</div>',
                unsafe_allow_html=True)

    if not has_data:
        st.info("Not enough data for model comparison.")
    else:
        if st.button("📈 Compare All Models", type="primary", use_container_width=True):
            with st.spinner("Training 3 models..."):
                comparison, forecasts = compare_all_models(
                    daily_df, holidays_df, forecast_days)
                st.session_state['comparison'] = comparison
                st.session_state['model_forecasts'] = forecasts

        if 'comparison' in st.session_state:
            comp   = st.session_state['comparison']
            fcasts = st.session_state['model_forecasts']

            # Metrics table
            if comp:
                comp_df = pd.DataFrame(comp).T.reset_index()
                comp_df.columns = ['Model','MSE','RMSE','MAE','R²','MAPE %']

                # Highlight best
                best_model = comp_df.loc[comp_df['RMSE'].idxmin(), 'Model']
                st.success(f"🏆 Best Model by RMSE: **{best_model}**")
                st.dataframe(comp_df.round(3), use_container_width=True, hide_index=True)

                # RMSE comparison chart
                fig_comp = go.Figure(go.Bar(
                    x=comp_df['Model'], y=comp_df['RMSE'],
                    marker_color=COLORS[:len(comp_df)],
                    text=comp_df['RMSE'].round(2), textposition='outside'))
                fig_comp.update_layout(**layout(height=300,
                    title='RMSE Comparison (lower = better)',
                    yaxis=dict(gridcolor='#27272a', title='RMSE (Rs)')))
                st.plotly_chart(fig_comp, use_container_width=True)

            # Forecast overlay chart
            if fcasts:
                st.markdown('<div class="section-title">All Model Forecasts Overlaid</div>',
                            unsafe_allow_html=True)
                fig_ov = go.Figure()
                fig_ov.add_trace(go.Bar(
                    x=daily_df['ds'], y=daily_df['y'],
                    name='Actual', marker_color='#3b82f6', opacity=0.5))
                for i, (model_name, fc) in enumerate(fcasts.items()):
                    fig_ov.add_trace(go.Scatter(
                        x=fc['ds'], y=fc['yhat'],
                        name=model_name,
                        line=dict(color=COLORS[i+1], width=2.5, dash='solid')))
                fig_ov.update_layout(**layout(height=350,
                    xaxis=dict(gridcolor='#27272a'),
                    yaxis=dict(gridcolor='#27272a', title='Rs'),
                    legend=dict(orientation='h', y=1.1)))
                st.plotly_chart(fig_ov, use_container_width=True)

                # Why RMSE explanation
                st.markdown('<div class="section-title">Why RMSE is used to pick the winner</div>',
                            unsafe_allow_html=True)
                st.markdown("""
| Metric | Why NOT the selector |
|---|---|
| **MSE** | Same info as RMSE but in Rs² — unreadable unit |
| **R²** | Goes negative with small data — unreliable |
| **MAE** | Treats all errors equally — misses spike days |
| **MAPE** | Breaks when actual values are very small |
| **RMSE ✅** | In rupees + penalizes big errors + always positive |
""")

# ══════════════════════════════════════════════════════
# TAB 6 — ANOMALIES
# ══════════════════════════════════════════════════════
with tab_anomaly:
    st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)

    if not has_data:
        st.info("Not enough data for anomaly detection.")
    else:
        # Auto-run forecast if not already in session
        if 'forecast' not in st.session_state:
            with st.spinner("⚙️ Auto-running forecast for anomaly detection..."):
                try:
                    _, auto_forecast = run_prophet(
                        daily_df, holidays_df, forecast_days,
                        cp_scale, sp_scale, seas_mode)
                    st.session_state['forecast'] = auto_forecast
                    st.success("✅ Forecast auto-generated!")
                except Exception as e:
                    st.error(f"Could not auto-run forecast: {e}")
                    st.stop()

        forecast  = st.session_state['forecast']
        anom_df   = detect_anomalies(daily_df, forecast, sigma=anomaly_sigma)
        anomalies = anom_df[anom_df['anomaly']].copy()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Days Analyzed", len(anom_df))
        with c2:
            st.metric("Anomalies Detected", len(anomalies),
                      delta=f"at {anomaly_sigma}σ threshold")
        with c3:
            if len(anomalies) > 0:
                worst = anomalies.loc[anomalies['severity'].idxmax()]
                st.metric("Worst Day Severity", "{:.1f}σ".format(worst["severity"]),
                          delta=str(worst['ds'].date()))

        # Residual chart
        st.markdown('<div class="section-title">Residuals — Actual vs Predicted</div>',
                    unsafe_allow_html=True)
        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(
            x=anom_df['ds'], y=anom_df['residual'],
            marker_color=['#ef4444' if a else '#3b82f6' for a in anom_df['anomaly']],
            name='Residual'))
        fig_res.add_hline(y=anomaly_sigma * anom_df['residual'].std(),
                          line_dash='dash', line_color='#f97316',
                          annotation_text=f'+{anomaly_sigma}σ')
        fig_res.add_hline(y=-anomaly_sigma * anom_df['residual'].std(),
                          line_dash='dash', line_color='#f97316',
                          annotation_text=f'-{anomaly_sigma}σ')
        fig_res.update_layout(**layout(height=280,
            yaxis=dict(gridcolor='#27272a', title='Residual Rs')))
        st.plotly_chart(fig_res, use_container_width=True)

        # Anomaly cards with AI explanations
        if len(anomalies) > 0:
            st.markdown('<div class="section-title">Flagged Days</div>',
                        unsafe_allow_html=True)
            for _, row in anomalies.iterrows():
                direction = "↑ overspent" if row['residual'] < 0 else "↓ underspent"
                anom_date = str(row['ds'].date())
                actual_val = round(row['y'])
                pred_val = round(row['yhat'])
                sev_val = round(row['severity'], 1)
                card_html = (
                    '<div class="anomaly-card">'
                    '<div class="anomaly-date">'
                    f'📅 {anom_date} &nbsp;|&nbsp; '
                    f'Actual: Rs {actual_val} &nbsp;|&nbsp; '
                    f'Expected: Rs {pred_val} &nbsp;|&nbsp; '
                    f'Severity: {sev_val}σ &nbsp;|&nbsp; {direction}'
                    '</div></div>'
                )
                st.markdown(card_html, unsafe_allow_html=True)

            # AI explanations for all anomalies
            if st.button("🤖 Get AI Explanation for All Anomalies", type="primary"):
                for _, row in anomalies.iterrows():
                    with st.spinner("Explaining {}...".format(row["ds"].date())):
                        cat_row = expenses_df[
                            expenses_df['date'].dt.date == row['ds'].date()]
                        cat = cat_row['category'].mode()[0] \
                              if not cat_row.empty else "General"
                        explanation = explain_anomaly(
                            str(row['ds'].date()), row['y'],
                            row['yhat'], cat, row['severity'])
                    exp_date = str(row['ds'].date())
                    exp_html = (
                        '<div class="anomaly-card">'
                        f'<div class="anomaly-date">📅 {exp_date} — {cat}</div>'
                        f'<div class="anomaly-explain">{explanation}</div>'
                        '</div>'
                    )
                    st.markdown(exp_html, unsafe_allow_html=True)
        else:
            st.success(f"✅ No anomalies detected at {anomaly_sigma}σ threshold.")

# ══════════════════════════════════════════════════════
# TAB 7 — AI ADVISOR
# ══════════════════════════════════════════════════════
with tab_ai:
    st.markdown('<div class="section-title">🤖 Claude AI Spending Advisor</div>',
                unsafe_allow_html=True)
    st.caption("Powered by Claude — get personalized, data-driven financial advice.")

    if not has_data:
        st.info("Add some expenses first to get personalized advice.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("💡 Get Personalized Advice", type="primary",
                         use_container_width=True):
                cat_totals   = dict(cat_df.groupby('Category')['Amount'].sum())
                month_spend  = get_current_month_spending()
                budgets_now  = get_budgets()
                budget_status = {}

                if not budgets_now.empty:
                    for _, brow in budgets_now.iterrows():
                        cat   = brow['category']
                        limit = brow['monthly_limit']
                        spent_row = month_spend[month_spend['category'] == cat]
                        spent     = float(spent_row['spent'].iloc[0]) \
                                    if not spent_row.empty else 0
                        budget_status[cat] = {
                            'spent': spent, 'limit': limit,
                            'pct'  : spent / limit * 100 if limit > 0 else 0
                        }

                # Auto-run forecast if not yet done
                if 'forecast' not in st.session_state:
                    with st.spinner("⚙️ Auto-running forecast..."):
                        try:
                            _, auto_fc = run_prophet(
                                daily_df, holidays_df, forecast_days,
                                cp_scale, sp_scale, seas_mode)
                            st.session_state['forecast'] = auto_fc
                        except Exception:
                            pass

                anom_count = 0
                if 'forecast' in st.session_state:
                    fc      = st.session_state['forecast']
                    anom_df = detect_anomalies(daily_df, fc, anomaly_sigma)
                    anom_count = int(anom_df['anomaly'].sum())

                forecast_total = 0
                if 'forecast' in st.session_state:
                    fc       = st.session_state['forecast']
                    fut      = fc[fc['ds'] > daily_df['ds'].max()]
                    forecast_total = fut['yhat'].sum()

                stats = get_summary_stats()

                with st.spinner("Claude is analyzing your spending..."):
                    advice = get_spending_advice(
                        cat_totals, budget_status,
                        anom_count, forecast_total,
                        stats.get('avg_daily', 0))

                st.session_state['ai_advice'] = advice

            if 'ai_advice' in st.session_state:
                st.markdown(
                    f'<div class="ai-box">{st.session_state["ai_advice"]}</div>',
                    unsafe_allow_html=True)

        with col2:
            st.markdown("**Weekly Comparison**")
            if st.button("📊 Compare This Week vs Last Week"):
                today = pd.Timestamp.today()
                week_start = today - timedelta(days=today.weekday())
                last_week_start = week_start - timedelta(days=7)

                this_week_df = expenses_df[
                    expenses_df['date'] >= week_start].groupby('category')['amount'].sum()
                last_week_df = expenses_df[
                    (expenses_df['date'] >= last_week_start) &
                    (expenses_df['date'] < week_start)].groupby('category')['amount'].sum()

                if len(this_week_df) > 0 and len(last_week_df) > 0:
                    with st.spinner("Analyzing weekly patterns..."):
                        summary = get_weekly_summary(
                            this_week_df.to_dict(), last_week_df.to_dict())
                    st.markdown(f'<div class="ai-box">{summary}</div>',
                                unsafe_allow_html=True)
                else:
                    st.info("Need data from both weeks for comparison.")

# ══════════════════════════════════════════════════════
# TAB 8 — REPORTS
# ══════════════════════════════════════════════════════
with tab_report:
    st.markdown('<div class="section-title">Weekly & Monthly Summary</div>',
                unsafe_allow_html=True)

    if not has_data:
        st.info("Not enough data for reports.")
    else:
        # Monthly report
        monthly = expenses_df.copy()
        monthly['month'] = monthly['date'].dt.to_period('M').astype(str)
        monthly_sum = monthly.groupby(['month','category'])['amount'].sum().reset_index()

        fig_monthly = px.bar(monthly_sum, x='month', y='amount', color='category',
                             color_discrete_sequence=COLORS, barmode='stack',
                             title='Monthly Spending by Category')
        fig_monthly.update_layout(**layout(height=350,
            xaxis=dict(gridcolor='#27272a'),
            yaxis=dict(gridcolor='#27272a', title='Rs'),
            legend=dict(orientation='h', y=1.1)))
        st.plotly_chart(fig_monthly, use_container_width=True)

        # Weekly report
        weekly = expenses_df.copy()
        weekly['week'] = weekly['date'].dt.to_period('W').astype(str)
        weekly_sum = weekly.groupby('week')['amount'].sum().reset_index()

        fig_weekly = go.Figure(go.Bar(
            x=weekly_sum['week'], y=weekly_sum['amount'],
            marker_color='#3b82f6', text=weekly_sum['amount'].round(0),
            textposition='outside'))
        fig_weekly.update_layout(**layout(height=280,
            title='Weekly Total Spending',
            xaxis=dict(tickangle=30, gridcolor='#27272a'),
            yaxis=dict(gridcolor='#27272a', title='Rs')))
        st.plotly_chart(fig_weekly, use_container_width=True)

        # Payment mode breakdown
        c1, c2 = st.columns(2)
        with c1:
            pay_sum = expenses_df.groupby('payment_mode')['amount'].sum().reset_index()
            fig_pay = px.pie(pay_sum, values='amount', names='payment_mode',
                             title='Spending by Payment Mode',
                             color_discrete_sequence=COLORS)
            fig_pay.update_layout(**layout(height=280, margin_t=40))
            st.plotly_chart(fig_pay, use_container_width=True)

        with c2:
            day_sum = expenses_df.copy()
            day_sum['day'] = day_sum['date'].dt.day_name()
            day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
            day_totals = day_sum.groupby('day')['amount'].mean().reindex(day_order).reset_index()
            fig_day = go.Figure(go.Bar(
                x=day_totals['day'], y=day_totals['amount'],
                marker_color='#8b5cf6'))
            fig_day.update_layout(**layout(height=280,
                title='Avg Spending by Day of Week',
                yaxis=dict(gridcolor='#27272a', title='Avg Rs')))
            st.plotly_chart(fig_day, use_container_width=True)

# ══════════════════════════════════════════════════════
# TAB 9 — EXPORT
# ══════════════════════════════════════════════════════
with tab_export:
    st.markdown('<div class="section-title">Export Full Report to Excel</div>',
                unsafe_allow_html=True)

    if not has_data:
        st.info("Add expenses first.")
    else:
        if st.button("📊 Generate Excel Report", type="primary", use_container_width=True):
            # Auto-run forecast if not yet done
            if 'forecast' not in st.session_state:
                with st.spinner("⚙️ Auto-running forecast..."):
                    try:
                        _, auto_fc = run_prophet(
                            daily_df, holidays_df, forecast_days,
                            cp_scale, sp_scale, seas_mode)
                        st.session_state['forecast'] = auto_fc
                    except Exception:
                        pass
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                # Sheet 1: All expenses
                expenses_df.to_excel(writer, sheet_name='All Expenses', index=False)

                # Sheet 2: Category summary
                cat_df.to_excel(writer, sheet_name='Category Summary', index=False)

                # Sheet 3: Daily totals
                daily_df.to_excel(writer, sheet_name='Daily Totals', index=False)

                # Sheet 4: Budgets
                if not budgets_df.empty:
                    budgets_df.to_excel(writer, sheet_name='Budgets', index=False)

                # Sheet 5: Forecast
                if 'forecast' in st.session_state:
                    fc = st.session_state['forecast']
                    fut = fc[fc['ds'] > daily_df['ds'].max()]
                    fut[['ds','yhat','yhat_lower','yhat_upper']].rename(columns={
                        'ds':'Date','yhat':'Forecast Rs',
                        'yhat_lower':'Lower Rs','yhat_upper':'Upper Rs'
                    }).to_excel(writer, sheet_name='Forecast', index=False)

                # Sheet 6: Anomalies
                if 'forecast' in st.session_state:
                    anom = detect_anomalies(
                        daily_df, st.session_state['forecast'], anomaly_sigma)
                    anom[anom['anomaly']].to_excel(
                        writer, sheet_name='Anomalies', index=False)

                # Sheet 7: Model comparison
                if 'comparison' in st.session_state:
                    pd.DataFrame(st.session_state['comparison']).T.reset_index(
                        ).rename(columns={'index':'Model'}).to_excel(
                        writer, sheet_name='Model Comparison', index=False)

            buf.seek(0)
            st.download_button(
                label="⬇️ Download Full Report (.xlsx)",
                data=buf,
                file_name=f"expense_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
            st.caption("Sheets: All Expenses · Category Summary · Daily Totals · "
                       "Budgets · Forecast · Anomalies · Model Comparison")

    # Active config table
    st.markdown('<div class="section-title">Active Configuration</div>',
                unsafe_allow_html=True)
    config_df = pd.DataFrame({
        'Parameter' : ['Forecast Days','Changepoint Prior','Seasonality Prior',
                       'Seasonality Mode','Anomaly Threshold','Spike Alert'],
        'Value'     : [forecast_days, cp_scale, sp_scale,
                       seas_mode, f'{anomaly_sigma}σ', f'Rs {spike_threshold}'],
    })
    st.dataframe(config_df, use_container_width=True, hide_index=True)

    if 'best_params' in st.session_state:
        bp = st.session_state['best_params']
        st.success("Auto-tuned params active — CP={} SP={} Mode={}".format(bp["cp"], bp["sp"], bp["mode"]))
