"""
split_data_advanced.py
======================
STEP 1 (Advanced) — Data Loading + SQLite Seeding

What this script does:
  1. Loads student_data.xlsx + excel_excel_xlsx.xlsx
  2. Cleans and validates data
  3. Seeds data into SQLite database (expense_tracker.db)
  4. Also creates legacy Excel files for backward compatibility

Run this FIRST before advanced_app.py
"""

import pandas as pd
import numpy as np
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database import init_db, seed_from_excel, get_summary_stats

print("=" * 60)
print("  STEP 1 (Advanced) — Data Loading & Database Setup")
print("=" * 60)

# ──────────────────────────────────────────────
# 1. INITIALIZE DATABASE
# ──────────────────────────────────────────────
print("\n🗄️  Initializing SQLite database...")
init_db()
print("   ✅ Database initialized → expense_tracker.db")

# ──────────────────────────────────────────────
# 2. SEED FROM EXCEL → DATABASE
# ──────────────────────────────────────────────
print("\n📂 Loading and seeding Excel data into database...")
success, msg = seed_from_excel()
if success:
    print(f"   ✅ {msg}")
else:
    print(f"   ❌ {msg}")
    print("   ℹ️  Make sure student_data.xlsx is in the same folder")

# ──────────────────────────────────────────────
# 3. VERIFY DATABASE
# ──────────────────────────────────────────────
print("\n📊 Verifying database contents...")
stats = get_summary_stats()
if stats:
    print(f"   ✅ Total Records  : {stats['total_records']}")
    print(f"   ✅ Total Spend    : Rs {stats['total_spend']:,.2f}")
    print(f"   ✅ Date Range     : {stats['date_range_start'].date()} → {stats['date_range_end'].date()}")
    print(f"   ✅ Categories     : {stats['categories']}")
    print(f"   ✅ Avg Daily Spend: Rs {stats['avg_daily']:.2f}")
    print(f"   ✅ Top Category   : {stats['top_category']}")
else:
    print("   ⚠️  No data in database yet")

# ──────────────────────────────────────────────
# 4. ALSO CREATE LEGACY EXCEL FILES
# ──────────────────────────────────────────────
print("\n📁 Creating legacy Excel files for backward compatibility...")

try:
    from database import get_daily_totals, get_category_grouped, get_holidays, get_all_expenses

    daily_df = get_daily_totals()
    if not daily_df.empty:
        daily_df.rename(columns={'ds':'Date','y':'Amount'}).to_excel("daily_total.xlsx", index=False)
        print("   ✅ daily_total.xlsx")

    cat_df = get_category_grouped()
    if not cat_df.empty:
        cat_df.to_excel("category_grouped.xlsx", index=False)
        print("   ✅ category_grouped.xlsx")

    holidays = get_holidays()
    if not holidays.empty:
        holidays.to_excel("holidays.xlsx", index=False)
        print(f"   ✅ holidays.xlsx ({len(holidays)} events)")
    else:
        pd.DataFrame(columns=['ds','holiday','lower_window','upper_window']).to_excel(
            "holidays.xlsx", index=False)
        print("   ✅ holidays.xlsx (empty — no events in data)")

    # Split by category
    all_exp = get_all_expenses()
    if not all_exp.empty:
        os.makedirs("category_files", exist_ok=True)
        for cat, grp in all_exp.groupby("category"):
            grp.to_excel(f"category_files/{cat}.xlsx", index=False)
            print(f"   ✅ category_files/{cat}.xlsx → {len(grp)} rows")

except Exception as e:
    print(f"   ⚠️  Could not create legacy files: {e}")

# ──────────────────────────────────────────────
# 5. SUMMARY
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  📋 SETUP COMPLETE")
print("=" * 60)
print("  Files Created:")
print("     ✅ expense_tracker.db   ← Main database (SQLite)")
print("     ✅ daily_total.xlsx     ← Legacy compatibility")
print("     ✅ category_grouped.xlsx")
print("     ✅ holidays.xlsx")
print()
print("  ▶️  Next → Run: streamlit run advanced_app.py")
print("=" * 60)
