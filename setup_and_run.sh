#!/bin/bash
echo "============================================"
echo "  Smart Expense Forecaster - Setup"
echo "============================================"

echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo ""
echo "Step 2: Installing dependencies..."
pip install -r requirements_advanced.txt

echo ""
echo "Step 3: Setting up database..."
python split_data_advanced.py

echo ""
echo "Step 4: Running category forecasting..."
python category_advanced.py

echo ""
echo "Step 5: Running overall forecasting..."
python prophet_advanced.py

echo ""
echo "============================================"
echo "  Setup Complete! Launching dashboard..."
echo "============================================"
streamlit run advanced_app.py
