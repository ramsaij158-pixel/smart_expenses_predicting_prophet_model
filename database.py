"""
database.py
===========
SQLite Database Handler
Replaces Excel file storage with a proper database.
Supports multiple users, real-time data entry, and budget management.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = "expense_tracker.db"

# ──────────────────────────────────────────────
# CONNECTION
# ──────────────────────────────────────────────
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ──────────────────────────────────────────────
# INITIALIZE DATABASE
# ──────────────────────────────────────────────
def init_db():
    conn = get_connection()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS expenses (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            date          TEXT    NOT NULL,
            amount        REAL    NOT NULL,
            category      TEXT    NOT NULL,
            description   TEXT    DEFAULT '',
            payment_mode  TEXT    DEFAULT 'Cash',
            event         TEXT    DEFAULT '',
            created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS budgets (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            category      TEXT    NOT NULL UNIQUE,
            monthly_limit REAL    NOT NULL,
            updated_at    TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS forecast_cache (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            category      TEXT    NOT NULL,
            forecast_date TEXT    NOT NULL,
            predicted     REAL,
            lower_bound   REAL,
            upper_bound   REAL,
            model_used    TEXT    DEFAULT 'Prophet',
            created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

# ──────────────────────────────────────────────
# EXPENSE CRUD
# ──────────────────────────────────────────────
def add_expense(date, amount, category, description="", payment_mode="Cash", event=""):
    conn = get_connection()
    conn.execute('''
        INSERT INTO expenses (date, amount, category, description, payment_mode, event)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (str(date), amount, category, description, payment_mode, event))
    conn.commit()
    conn.close()

def get_all_expenses():
    conn = get_connection()
    df = pd.read_sql_query('SELECT * FROM expenses ORDER BY date DESC', conn)
    conn.close()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def get_expenses_by_category(category):
    conn = get_connection()
    df = pd.read_sql_query(
        'SELECT * FROM expenses WHERE category = ? ORDER BY date', conn, params=(category,))
    conn.close()
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

def get_daily_totals():
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT date as ds, SUM(amount) as y
        FROM expenses
        GROUP BY date
        ORDER BY date
    ''', conn)
    conn.close()
    if not df.empty:
        df['ds'] = pd.to_datetime(df['ds'])
    return df

def get_category_grouped():
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT date as Date, category as Category, SUM(amount) as Amount
        FROM expenses
        GROUP BY date, category
        ORDER BY date, category
    ''', conn)
    conn.close()
    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_holidays():
    conn = get_connection()
    df = pd.read_sql_query('''
        SELECT DISTINCT date as ds, event as holiday
        FROM expenses
        WHERE event != '' AND event IS NOT NULL
    ''', conn)
    conn.close()
    if not df.empty:
        df['ds'] = pd.to_datetime(df['ds'])
        df['lower_window'] = 0
        df['upper_window'] = 1
    return df

def delete_expense(expense_id):
    conn = get_connection()
    conn.execute('DELETE FROM expenses WHERE id = ?', (expense_id,))
    conn.commit()
    conn.close()

def update_expense(expense_id, date, amount, category, description, payment_mode, event):
    conn = get_connection()
    conn.execute('''
        UPDATE expenses
        SET date=?, amount=?, category=?, description=?, payment_mode=?, event=?
        WHERE id=?
    ''', (str(date), amount, category, description, payment_mode, event, expense_id))
    conn.commit()
    conn.close()

# ──────────────────────────────────────────────
# BUDGET CRUD
# ──────────────────────────────────────────────
def set_budget(category, monthly_limit):
    conn = get_connection()
    conn.execute('''
        INSERT OR REPLACE INTO budgets (category, monthly_limit, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    ''', (category, monthly_limit))
    conn.commit()
    conn.close()

def get_budgets():
    conn = get_connection()
    df = pd.read_sql_query('SELECT * FROM budgets', conn)
    conn.close()
    return df

def get_current_month_spending():
    conn = get_connection()
    current_month = datetime.now().strftime('%Y-%m')
    df = pd.read_sql_query('''
        SELECT category, SUM(amount) as spent
        FROM expenses
        WHERE strftime('%Y-%m', date) = ?
        GROUP BY category
    ''', conn, params=(current_month,))
    conn.close()
    return df

# ──────────────────────────────────────────────
# SEED FROM EXCEL
# ──────────────────────────────────────────────
def seed_from_excel():
    """Load existing Excel data into SQLite on first run"""
    conn = get_connection()
    count = conn.execute('SELECT COUNT(*) FROM expenses').fetchone()[0]
    conn.close()

    if count > 0:
        return True, f"Database already has {count} records"

    try:
        dfs = []
        for fname in ["student_data.xlsx", "excel_excel_xlsx.xlsx"]:
            if os.path.exists(fname):
                dfs.append(pd.read_excel(fname))

        if not dfs:
            return False, "No Excel files found"

        df = pd.concat(dfs, ignore_index=True)
        df['Date'] = pd.to_datetime(df['Date'])

        conn = get_connection()
        for _, row in df.iterrows():
            event = str(row.get('Event', '')) if pd.notna(row.get('Event', '')) else ''
            conn.execute('''
                INSERT INTO expenses (date, amount, category, description, event)
                VALUES (?, ?, ?, ?, ?)
            ''', (row['Date'].strftime('%Y-%m-%d'), row['Amount'],
                  row['Category'], '', event))
        conn.commit()
        conn.close()
        return True, f"Seeded {len(df)} records from Excel"
    except Exception as e:
        return False, str(e)

# ──────────────────────────────────────────────
# STATS HELPERS
# ──────────────────────────────────────────────
def get_summary_stats():
    conn = get_connection()
    df = pd.read_sql_query('SELECT * FROM expenses', conn)
    conn.close()
    if df.empty:
        return {}
    df['date'] = pd.to_datetime(df['date'])
    return {
        'total_spend'     : df['amount'].sum(),
        'avg_daily'       : df.groupby('date')['amount'].sum().mean(),
        'max_day'         : df.groupby('date')['amount'].sum().max(),
        'total_records'   : len(df),
        'categories'      : df['category'].nunique(),
        'date_range_start': df['date'].min(),
        'date_range_end'  : df['date'].max(),
        'top_category'    : df.groupby('category')['amount'].sum().idxmax(),
    }
