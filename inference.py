"""
inference.py

Baseline inference script for the Financial Fraud Detection environment.
Runs an LLM agent against all 3 tasks and reports scores.

REQUIRED by hackathon:
  - Must be named inference.py at root
  - Must use OpenAI client
  - Must read from env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN
  - Must complete in under 20 minutes
  - Must run on vcpu=2, memory=8gb

Usage:
    export API_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
    export MODEL_NAME="gemma-3-27b-it"
    export HF_TOKEN="your-api-key"
    python inference.py
"""

import os
import sys
import json
import time

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ff_env"))

from ff_env.server.ff_env_environment import FfEnvironment
from ff_env.models import FraudAction

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "dummy")

TASK_SEEDS = {"easy": 1020790, "medium": 1687950, "hard": 1245620}

# ─────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a forensic accountant auditing a company's financial statements.
You will receive quarterly financial data across 3 statements:
  - Income Statement (revenue, COGS, gross profit, operating expenses, net income)
  - Balance Sheet (cash, receivables, inventory, total assets, liabilities, equity)
  - Cash Flow Statement (operating, investing, financing cash flows)

Your job is to detect financial fraud by taking investigative actions.

AVAILABLE ACTIONS — respond with ONLY valid JSON, no markdown, no explanation:

1. {"action_type": "inspect", "parameters": {"statement": "income_statement", "metric": "revenue"}}
   Valid statements: income_statement, balance_sheet, cash_flow
   Valid metrics -- income_statement: revenue, cost_of_goods_sold, gross_profit, operating_expenses, net_income
                    balance_sheet: cash, receivables, inventory, total_assets, total_liabilities, equity
                    cash_flow: operating, investing, financing, net_change

2. {"action_type": "compare", "parameters": {"metric_a": "revenue", "metric_b": "operating_cashflow"}}

3. {"action_type": "check_benchmark", "parameters": {"metric": "receivables_ratio"}}
   Valid: receivables_ratio, gross_margin, asset_growth

4. {"action_type": "request_detail", "parameters": {"quarter": 2}}
   (0=Q1, 1=Q2, 2=Q3, 3=Q4)

5. {"action_type": "flag", "parameters": {"fraud_type": "revenue_inflation", "line_item": "revenue", "statement": "income_statement", "quarters": [2, 3]}}
   Valid fraud types: revenue_inflation, expense_hiding, channel_stuffing, earnings_smoothing, asset_overstatement

6. {"action_type": "submit_report", "parameters": {}}

STRATEGY:
- Inspect revenue and net_income first
- Compare revenue vs operating_cashflow -- divergence is a red flag
- Check receivables growth vs revenue
- Flag ALL frauds before submitting
- False positives hurt your score

Respond with ONLY the JSON. Nothing else."""


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def build_user_message(obs) -> str:
    return f"""COMPANY: {obs.company_name} | INDUSTRY: {obs.industry}
QUARTERS: {obs.quarters}
TASK: {obs.task_name.upper()} -- {obs.task_description}
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

=== CASH FLOW ($K) ===
Operating CF:       {obs.operating_cashflow}
Investing CF:       {obs.investing_cashflow}
Financing CF:       {obs.financing_cashflow}
Net Change:         {obs.net_cash_change}

=== BENCHMARKS ===
Gross Margin:       {obs.benchmark_gross_margin}
Receivables Ratio:  {obs.benchmark_receivables_ratio}

=== FLAGS RAISED SO FAR ===
{obs.flags_raised if obs.flags_raised else "None yet"}

=== LAST RESULT ===
{obs.last_action_result}

Respond with ONLY a JSON action object."""


def build_messages(history: list) -> list:
    """
    Build messages for API call.
    Merges system prompt into first user message for compatibility
    with models that don't support system role (e.g. Gemma).
    """
    if not history:
        return [{"role": "user", "content": SYSTEM_PROMPT}]
    merged_first = {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + history[0]["content"]}
    return [merged_first] + history[1:]


def call_llm(client: OpenAI, history: list) -> str | None:
    """Call LLM with retry on rate limit."""
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=build_messages(history),
                max_tokens=200,
                temperature=0.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 15 * (attempt + 1)
                print(f"    Rate limited -- waiting {wait}s (attempt {attempt+1}/6)")
                time.sleep(wait)
            else:
                print(f"    LLM error: {e}")
                return None
    return None


# ─────────────────────────────────────────────────────────────
# Agent Loop
# ─────────────────────────────────────────────────────────────

def run_agent(client: OpenAI, env: FfEnvironment, task_name: str, seed: int) -> dict:
    print(f"\n{'='*55}")
    print(f"TASK: {task_name.upper()}  (seed={seed})")
    print(f"{'='*55}")

    obs = env.reset(task_name=task_name, seed=seed)
    history = []
    trajectory = []
    final_score = 0.0
    step = 0

    while not obs.done:
        step += 1
        user_msg = build_user_message(obs)
        history.append({"role": "user", "content": user_msg})

        raw_action = call_llm(client, history)
        if raw_action is None:
            print(f"  [Step {step}] LLM failed -- ending episode")
            break

        history.append({"role": "assistant", "content": raw_action})

        # Parse JSON action
        try:
            clean = raw_action.replace("```json", "").replace("```", "").strip()
            action_dict = json.loads(clean)
            action = FraudAction(
                action_type=action_dict.get("action_type", "submit_report"),
                parameters=action_dict.get("parameters", {}),
            )
        except Exception as e:
            print(f"  [Step {step}] Parse error: {e} | Raw: {raw_action[:80]}")
            action = FraudAction(action_type="submit_report", parameters={})

        obs = env.step(action)
        print(f"  [Step {step}] {action.action_type:<20} reward={obs.reward:.3f}")

        trajectory.append({
            "step":        step,
            "action_type": action.action_type,
            "parameters":  action.parameters,
            "reward":      obs.reward,
            "done":        obs.done,
        })

        if obs.done:
            try:
                line = [l for l in obs.last_action_result.split("\n") if "Final score" in l]
                if line:
                    final_score = float(line[0].split(":")[1].strip().split("/")[0].strip())
            except Exception:
                final_score = obs.reward

        time.sleep(1.0)

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
    print("Financial Fraud Detection -- Baseline Inference")
    print(f"Model    : {MODEL_NAME}")
    print(f"API Base : {API_BASE_URL}")

    if not API_BASE_URL or not MODEL_NAME:
        print("ERROR: API_BASE_URL and MODEL_NAME must be set")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    env = FfEnvironment()
    results = []

    for task_name, seed in TASK_SEEDS.items():
        result = run_agent(client, env, task_name=task_name, seed=seed)
        results.append(result)

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

    output = {"model": MODEL_NAME, "results": results, "average": avg}
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to baseline_results.json")
    return avg


if __name__ == "__main__":
    main()