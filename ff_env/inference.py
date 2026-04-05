"""
inference.py

Baseline inference script for the Financial Fraud Detection environment.

MANDATORY stdout format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
 
Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
 
Usage:
    export API_BASE_URL="https://api.groq.com/openai/v1"
    export MODEL_NAME="llama-3.3-70b-versatile"
    export HF_TOKEN="your-api-key"
    python3 inference.py
    
"""

import os
import sys
import json
import time

from typing import Optional, List
from openai import OpenAI

# Works both locally and on HuggingFace (flat structure)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from server.ff_env_environment import FfEnvironment
    from models import FraudAction
except ImportError:
    from ff_env.server.ff_env_environment import FfEnvironment
    from ff_env.models import FraudAction



#Config

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "dummy")
BENCHMARK = "financial_fraud_detection"

TASKS = [
{"name": "easy", "seed": 1020790}, 
{"name": "medium", "seed": 1687950}, 
{"name": "hard", "seed": 1245620}
]

SUCCESS_SCORE_THRESHOLD = 0.7


#stdout logging 
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)
 
 
def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Truncate action string to keep line readable
    action_str = action.replace("\n", " ")[:120]
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )
 
 
def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )
 

#Prompt

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




#Helpers

def build_user_message(obs) -> str:
    return (
        f"COMPANY: {obs.company_name} | INDUSTRY: {obs.industry}\n"
        f"QUARTERS: {obs.quarters}\n"
        f"TASK: {obs.task_name.upper()} -- {obs.task_description}\n"
        f"STEPS REMAINING: {obs.step_budget}\n\n"
        f"=== INCOME STATEMENT ($K) ===\n"
        f"Revenue:            {obs.revenue}\n"
        f"Cost of Goods Sold: {obs.cost_of_goods_sold}\n"
        f"Gross Profit:       {obs.gross_profit}\n"
        f"Operating Expenses: {obs.operating_expenses}\n"
        f"Net Income:         {obs.net_income}\n\n"
        f"=== BALANCE SHEET ($K) ===\n"
        f"Cash:               {obs.cash}\n"
        f"Receivables:        {obs.receivables}\n"
        f"Inventory:          {obs.inventory}\n"
        f"Total Assets:       {obs.total_assets}\n"
        f"Total Liabilities:  {obs.total_liabilities}\n"
        f"Equity:             {obs.equity}\n\n"
        f"=== CASH FLOW ($K) ===\n"
        f"Operating CF:       {obs.operating_cashflow}\n"
        f"Investing CF:       {obs.investing_cashflow}\n"
        f"Financing CF:       {obs.financing_cashflow}\n"
        f"Net Change:         {obs.net_cash_change}\n\n"
        f"=== BENCHMARKS ===\n"
        f"Gross Margin:       {obs.benchmark_gross_margin}\n"
        f"Receivables Ratio:  {obs.benchmark_receivables_ratio}\n\n"
        f"=== FLAGS RAISED SO FAR ===\n"
        f"{obs.flags_raised if obs.flags_raised else 'None yet'}\n\n"
        f"=== LAST RESULT ===\n"
        f"{obs.last_action_result}\n\n"
        f"Respond with ONLY a JSON action object."
    )


def build_messages(history: list) -> list:
    """Merge system prompt into first user message for model compatibility."""
    if not history:
        return [{"role": "user", "content": SYSTEM_PROMPT}]
    merged = {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + history[0]["content"]}
    return [merged] + history[1:]
 
 
def call_llm(client: OpenAI, history: list) -> Optional[str]:
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
                print(f"[DEBUG] Rate limited -- waiting {wait}s (attempt {attempt+1}/6)", flush=True)
                time.sleep(wait)
            else:
                print(f"[DEBUG] LLM error: {e}", flush=True)
                return None
    return None





#Run one task episode

def run_task(client: OpenAI, env: FfEnvironment, task_name: str, seed: int) -> dict:
    """Run one task and return results dict."""
 
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
 
    obs = env.reset(task_name=task_name, seed=seed)
    history    = []
    rewards    = []
    steps_taken = 0
    final_score = 0.0
    success     = False
 
    try:
        while not obs.done:
            steps_taken += 1
            user_msg = build_user_message(obs)
            history.append({"role": "user", "content": user_msg})
 
            raw_action = call_llm(client, history)
            if raw_action is None:
                log_step(steps_taken, "null", 0.0, True, "LLM call failed")
                rewards.append(0.0)
                break
 
            history.append({"role": "assistant", "content": raw_action})
 
            # Parse action
            error_msg = None
            try:
                clean = raw_action.replace("```json", "").replace("```", "").strip()
                action_dict = json.loads(clean)
                action = FraudAction(
                    action_type=action_dict.get("action_type", "submit_report"),
                    parameters=action_dict.get("parameters", {}),
                )
                action_str = f"{action.action_type}({json.dumps(action.parameters)})"
            except Exception as e:
                error_msg = f"parse_error:{str(e)[:50]}"
                action = FraudAction(action_type="submit_report", parameters={})
                action_str = "submit_report({})"
 
            # Step environment
            obs = env.step(action)
            reward = obs.reward
            rewards.append(reward)
 
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=obs.done,
                error=error_msg,
            )
 
            # Extract final score from submit result
            if obs.done:
                try:
                    lines = obs.last_action_result.split("\n")
                    score_line = [l for l in lines if "Final score" in l]
                    if score_line:
                        final_score = float(score_line[0].split(":")[1].strip().split("/")[0].strip())
                    else:
                        final_score = reward
                except Exception:
                    final_score = reward
 
            time.sleep(1.5)  # avoid rate limits
 
        success = final_score >= SUCCESS_SCORE_THRESHOLD
 
    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        success = False
 
    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=final_score,
            rewards=rewards,
        )
 
    return {
        "task":        task_name,
        "seed":        seed,
        "steps_used":  steps_taken,
        "final_score": final_score,
        "success":     success,
        "rewards":     rewards,
    }







#Main

def main():
    if not API_BASE_URL or not MODEL_NAME:
        print("[DEBUG] ERROR: API_BASE_URL and MODEL_NAME must be set", flush=True)
        sys.exit(1)
 
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    env    = FfEnvironment()
    results = []
 
    for task in TASKS:
        result = run_task(client, env, task_name=task["name"], seed=task["seed"])
        results.append(result)
 
    #Save results
    avg = sum(r["final_score"] for r in results) / len(results)
    output = {"model": MODEL_NAME, "results": results, "average": avg}
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
 
    print(f"[DEBUG] Average score: {avg:.4f}", flush=True)
    print(f"[DEBUG] Results saved to baseline_results.json", flush=True)
 
 
if __name__ == "__main__":
    main()