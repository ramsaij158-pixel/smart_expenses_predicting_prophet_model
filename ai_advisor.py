"""
ai_advisor.py
=============
Claude API Integration for Smart Spending Advice
Provides: Personalized tips, Anomaly explanations, Budget recommendations
"""

import requests
import json

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
MODEL          = "claude-sonnet-4-20250514"

# ──────────────────────────────────────────────
# CORE API CALL
# ──────────────────────────────────────────────
def _call_claude(prompt: str, max_tokens: int = 600) -> str:
    try:
        resp = requests.post(
            CLAUDE_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model"     : MODEL,
                "max_tokens": max_tokens,
                "messages"  : [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        data = resp.json()
        if "content" in data:
            return data["content"][0]["text"]
        return f"API error: {data.get('error', {}).get('message', 'Unknown error')}"
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Could not connect to AI advisor: {str(e)}"

# ──────────────────────────────────────────────
# PERSONALIZED SPENDING ADVICE
# ──────────────────────────────────────────────
def get_spending_advice(category_totals: dict, budget_status: dict,
                        anomaly_count: int, forecast_total: float,
                        avg_daily: float) -> str:
    cat_lines = "\n".join(
        f"  - {cat}: Rs {amt:.0f}" for cat, amt in category_totals.items())

    budget_lines = ""
    if budget_status:
        lines = []
        for cat, row in budget_status.items():
            status = 'OVER BUDGET' if row['spent'] > row['limit'] else str(round(row['pct'])) + '% used'
            lines.append(f"  - {cat}: spent Rs {row['spent']:.0f} / budget Rs {row['limit']:.0f} ({status})")
        budget_lines = '\nBudget vs Actual:\n' + '\n'.join(lines)

    prompt = f"""You are a friendly personal finance advisor for a college student in India.

THEIR SPENDING DATA:
Category-wise totals:
{cat_lines}

Average daily spend: Rs {avg_daily:.0f}
{budget_lines}
Anomalies detected: {anomaly_count} unusual spending days
30-day forecast total: Rs {forecast_total:.0f}

Please provide:
1. **Top 3 Money-Saving Tips** — specific to their actual categories and amounts
2. **Biggest Concern** — which category needs attention and a practical fix
3. **One Positive** — something they are doing well
4. **Forecast Check** — is Rs {forecast_total:.0f} for next 30 days reasonable?

Keep it friendly, specific, and actionable. Use Rs for currency. Max 250 words.
"""
    return _call_claude(prompt, max_tokens=600)

# ──────────────────────────────────────────────
# ANOMALY EXPLANATION
# ──────────────────────────────────────────────
def explain_anomaly(date: str, actual: float, predicted: float,
                    category: str, severity: float) -> str:
    direction = "much higher" if actual > predicted else "much lower"
    diff      = abs(actual - predicted)

    prompt = f"""A student expense tracker flagged an unusual spending day.

Date: {date}
Category: {category}
Actual spending: Rs {actual:.0f}
Normal expected: Rs {predicted:.0f}
Difference: Rs {diff:.0f} ({direction} than expected)
Severity: {severity:.1f} standard deviations from normal

In exactly 2 sentences:
1. Explain what this anomaly means in simple terms for a student
2. Suggest the most likely cause and one action they can take

Be specific and practical. No jargon.
"""
    return _call_claude(prompt, max_tokens=150)

# ──────────────────────────────────────────────
# BUDGET RECOMMENDATION
# ──────────────────────────────────────────────
def get_budget_recommendation(category_totals: dict, months_of_data: float) -> str:
    monthly = {cat: amt / max(months_of_data, 0.5)
               for cat, amt in category_totals.items()}
    lines = "\n".join(
        f"  - {cat}: Rs {amt:.0f}/month on average"
        for cat, amt in monthly.items())

    prompt = f"""A college student wants help setting monthly spending budgets.

Their average monthly spending by category:
{lines}

Suggest a realistic monthly budget for each category that:
- Is 10-15% lower than their average (to encourage savings)
- Is practical for a student in India
- Adds up to a reasonable total

Format as a simple table:
Category | Current Avg | Suggested Budget | Savings

Keep it brief and encouraging. Use Rs for currency.
"""
    return _call_claude(prompt, max_tokens=400)

# ──────────────────────────────────────────────
# WEEKLY SPENDING SUMMARY
# ──────────────────────────────────────────────
def get_weekly_summary(this_week: dict, last_week: dict) -> str:
    def fmt(d):
        return "\n".join(f"  - {k}: Rs {v:.0f}" for k, v in d.items())

    prompt = f"""Compare a student's spending between two weeks and give a brief insight.

THIS WEEK:
{fmt(this_week)}
Total: Rs {sum(this_week.values()):.0f}

LAST WEEK:
{fmt(last_week)}
Total: Rs {sum(last_week.values()):.0f}

In 3 sentences: what changed, what drove the change, and one suggestion.
Be specific. Use Rs for currency.
"""
    return _call_claude(prompt, max_tokens=200)
