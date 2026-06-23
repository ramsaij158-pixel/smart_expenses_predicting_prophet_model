# Smart Expenses Predicting Prophet Model

An advanced student expense forecasting project that combines SQLite storage, category-wise analysis, Prophet forecasting, model comparison and a Streamlit dashboard.

## Project Goal

The goal of this project is to help students understand their spending patterns and forecast future expenses. It is designed as a practical machine-learning project for student finance.

## Key Features

- SQLite database for storing expense records
- Streamlit dashboard for interactive expense tracking
- Category-wise spending analysis
- Prophet-based time-series forecasting
- Linear Regression and Moving Average model comparison
- Walk-forward validation and model metrics
- Budget tracking and anomaly detection
- Excel report export
- AI spending-advice module structure

## Main Files

| File | Purpose |
|---|---|
| `advanced_app.py` | Streamlit dashboard for the full expense-forecasting app |
| `database.py` | SQLite database setup and expense CRUD operations |
| `models.py` | Forecasting models, metrics, validation and anomaly detection |
| `split_data_advanced.py` | Loads Excel data and seeds the SQLite database |
| `category_advanced.py` | Runs category-wise forecasting and model comparison |
| `prophet_advanced.py` | Runs overall forecasting, validation and reporting |
| `ai_advisor.py` | AI-advice helper functions for spending recommendations |
| `requirements_advanced.txt` | Python dependencies |

## Tech Stack

- Python
- Pandas and NumPy
- Prophet
- Scikit-learn
- SQLite
- Streamlit
- Plotly
- Excel files for input and output

## How to Run

Install dependencies:

```bash
pip install -r requirements_advanced.txt
```

Prepare the data:

```bash
python split_data_advanced.py
```

Run category forecasting:

```bash
python category_advanced.py
```

Run overall forecasting:

```bash
python prophet_advanced.py
```

Start the dashboard:

```bash
streamlit run advanced_app.py
```

## Learning Outcomes

- Built an end-to-end data science workflow
- Practiced time-series forecasting with Prophet
- Compared multiple forecasting approaches
- Used SQLite instead of only Excel files
- Created a dashboard for business-style presentation
- Explored how AI advice can be added to analytics projects

## Future Improvements

- Add authentication for multiple students
- Improve model evaluation with a larger dataset
- Add clearer visual reports and screenshots
- Store API keys securely through environment variables
- Convert the project into a deployable web app

