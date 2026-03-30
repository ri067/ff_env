"""
inference.py

Baseline inference script for the Financial Fraud Detection environment.
Runs an LLM agent against all 3 tasks and reports scores.

REQUIRED by hackathon:
  - Must be named inference.py
  - Must be in the root directory
  - Must use OpenAI client
  - Must read credentials from environment variables:
      API_BASE_URL  - the LLM API endpoint
      MODEL_NAME    - model identifier
      HF_TOKEN      - Hugging Face token
  - Must complete in under 20 minutes
  - Must run on vcpu=2, memory=8gb

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your-hf-token"
    python inference.py
"""

import os
import sys
import json
import time

from openai import OpenAI

# ── Add ff_env to path ────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ff_env"))

from ff_env.server.ff_env_environment import FfEnvironment
from ff_env.models import FraudAction


# ─────────────────────────────────────────────────────────────
# Config — read from environment variables
# ─────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

# Fixed seeds for reproducibility — same scenarios every run
TASK_SEEDS = {
    "easy":   42,
    "medium": 123,
    "hard":   999,
}


# ─────────────────────────────────────────────────────────────
# System Prompt
# Tells the LLM what it is and what actions it can take
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a forensic accountant auditing a company's financial statements.
You will receive quarterly financial data across 3 statements:
  - Income Statement (revenue, COGS, gross profit, operating expenses, net income)
  - Balance Sheet (cash, receivables, inventory, total assets, liabilities, equity)
  - Cash Flow Statement (operating, investing, financing cash flows)

Your job is to detect financial fraud by taking investigative actions.

AVAILABLE ACTIONS (respond with ONLY valid JSON):

1. Inspect a metric:
{"action_type": "inspect", "parameters": {"statement": "income_statement", "metric": "revenue"}}

Valid statements: income_statement, balance_sheet, cash_flow
Valid metrics:
  income_statement: revenue, cost_of_goods_sold, gross_profit, operating_expenses, net_income
  balance_sheet: cash, receivables, inventory, total_assets, total_liabilities, equity
  cash_flow: operating, investing, financing, net_change

2. Compare two metrics:
{"action_type": "compare", "parameters": {"metric_a": "revenue", "metric_b": "operating_cashflow"}}

3. Check industry benchmark:
{"action_type": "check_benchmark", "parameters": {"metric": "receivables_ratio"}}
Valid benchmark metrics: receivables_ratio, gross_margin, asset_growth

4. Request quarter detail:
{"action_type": "request_detail", "parameters": {"quarter": 2}}
(quarter: 0=Q1, 1=Q2, 2=Q3, 3=Q4)

5. Flag a fraud:
{"action_type": "flag", "parameters": {"fraud_type": "revenue_inflation", "line_item": "revenue", "statement": "income_statement", "quarters": [2, 3]}}

Valid fraud types: revenue_inflation, expense_hiding, channel_stuffing, earnings_smoothing, asset_overstatement

6. Submit your report (ends episode):
{"action_type": "submit_report", "parameters": {}}

STRATEGY:
- Start by inspecting key metrics (revenue, net_income, operating cash flow)
- Compare revenue vs operating cash flow — divergence is a major red flag
- Check receivables growth vs revenue growth
- Use benchmarks to spot unusual ratios
- Flag ALL frauds you find before submitting
- Don't submit until you've investigated thoroughly
- Avoid flagging things you're not confident about (false positives hurt your score)

Respond with ONLY the JSON action object. No explanation, no markdown, just JSON."""


# ─────────────────────────────────────────────────────────────
# Agent Loop
# ─────────────────────────────────────────────────────────────

def build_user_message(obs) -> str:
    """Convert observation to a readable message for the LLM."""
    return f"""COMPANY: {obs.company_name} | INDUSTRY: {obs.industry}
QUARTERS: {obs.quarters}
TASK: {obs.task_name.upper()} — {obs.task_description}
STEPS REMAINING: {obs.step_budget}

=== INCOME STATEMENT ($K) ===
Revenue:            {obs.revenue}
Cost of Goods Sold: {obs.cost_of_goods_sold}
Gross Profit:       {obs.gross_profit}
Operating Expenses: {obs.operating_expenses}
Net Income:         {obs.net_income}

=== BALANCE SHEET ($K) ===
Cash:               {obs.cash}
Receivables:        {obs.receivables}
Inventory:          {obs.inventory}
Total Assets:       {obs.total_assets}
Total Liabilities:  {obs.total_liabilities}
Equity:             {obs.equity}

=== CASH FLOW STATEMENT ($K) ===
Operating CF:       {obs.operating_cashflow}
Investing CF:       {obs.investing_cashflow}
Financing CF:       {obs.financing_cashflow}
Net Change:         {obs.net_cash_change}

=== INDUSTRY BENCHMARKS ===
Gross Margin:       {obs.benchmark_gross_margin}
Receivables Ratio:  {obs.benchmark_receivables_ratio}

=== YOUR FLAGS SO FAR ===
{obs.flags_raised if obs.flags_raised else "None yet"}

=== LAST ACTION RESULT ===
{obs.last_action_result}

What is your next action? Respond with ONLY a JSON action object."""


def run_agent(client: OpenAI, env: FfEnvironment, task_name: str, seed: int) -> dict:
    """
    Run the LLM agent on one task episode.
    Returns a results dict with score and trajectory.
    """
    print(f"\n{'='*55}")
    print(f"TASK: {task_name.upper()}  (seed={seed})")
    print(f"{'='*55}")

    obs = env.reset(task_name=task_name, seed=seed)
    conversation_history = []
    trajectory = []
    final_score = 0.0
    step = 0

    while not obs.done:
        step += 1
        user_msg = build_user_message(obs)
        conversation_history.append({"role": "user", "content": user_msg})

        # Call the LLM
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *conversation_history,
                ],
                max_tokens=200,
                temperature=0.0,  # deterministic
            )
            raw_action = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  [Step {step}] LLM error: {e}")
            break

        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": raw_action})

        # Parse action
        try:
            # Strip markdown fences if model added them
            clean = raw_action.replace("```json", "").replace("```", "").strip()
            action_dict = json.loads(clean)
            action = FraudAction(
                action_type=action_dict.get("action_type", "submit_report"),
                parameters=action_dict.get("parameters", {}),
            )
        except Exception as e:
            print(f"  [Step {step}] Parse error: {e} | Raw: {raw_action[:100]}")
            action = FraudAction(action_type="submit_report", parameters={})

        # Step environment
        obs = env.step(action)

        print(f"  [Step {step}] {action.action_type} → reward={obs.reward:.3f}")
        trajectory.append({
            "step":        step,
            "action_type": action.action_type,
            "parameters":  action.parameters,
            "reward":      obs.reward,
            "done":        obs.done,
        })

        # Extract final score from submit result
        if obs.done:
            result_text = obs.last_action_result
            try:
                score_line = [l for l in result_text.split("\n") if "Final score" in l]
                if score_line:
                    final_score = float(score_line[0].split(":")[1].strip().split("/")[0].strip())
            except Exception:
                final_score = obs.reward

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    print(f"  FINAL SCORE: {final_score:.4f}")

    return {
        "task":        task_name,
        "seed":        seed,
        "steps_used":  step,
        "final_score": final_score,
        "trajectory":  trajectory,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("Financial Fraud Detection — Baseline Inference")
    print(f"Model     : {MODEL_NAME}")
    print(f"API Base  : {API_BASE_URL}")

    # Validate env vars
    if not API_BASE_URL:
        print("ERROR: API_BASE_URL not set")
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME not set")
        sys.exit(1)

    # Init OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or "dummy-key",  # some endpoints don't need a key
    )

    env = FfEnvironment()
    results = []

    # Run all 3 tasks
    for task_name, seed in TASK_SEEDS.items():
        result = run_agent(client, env, task_name=task_name, seed=seed)
        results.append(result)

    # Summary
    print(f"\n{'='*55}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*55}")
    total = 0.0
    for r in results:
        print(f"  {r['task'].upper():<10} score={r['final_score']:.4f}  steps={r['steps_used']}")
        total += r["final_score"]

    avg = total / len(results)
    print(f"{'─'*55}")
    print(f"  AVERAGE    score={avg:.4f}")

    # Save results to file
    output = {
        "model":    MODEL_NAME,
        "results":  results,
        "average":  avg,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to baseline_results.json")

    return avg


if __name__ == "__main__":
    main()