"""
fraud_injector.py

Takes clean financial statements from data_generator.py and deliberately
injects specific fraud patterns into them.

Each injector function:
  1. Modifies specific numbers in the statements
  2. Records exactly what was changed in statements.injected_frauds
     (this is the ground truth the grader uses to score the agent)

The agent NEVER sees injected_frauds directly.
It must detect fraud by reasoning about the numbers.
"""

import random
from data_generator import FinancialStatements


# ─────────────────────────────────────────────────────────────
# Fraud Type Constants
# These strings are used in injected_frauds and by the grader
# to check if the agent identified the right fraud type.
# ─────────────────────────────────────────────────────────────

FRAUD_REVENUE_INFLATION    = "revenue_inflation"
FRAUD_EXPENSE_HIDING       = "expense_hiding"
FRAUD_CHANNEL_STUFFING     = "channel_stuffing"
FRAUD_EARNINGS_SMOOTHING   = "earnings_smoothing"
FRAUD_ASSET_OVERSTATEMENT  = "asset_overstatement"

ALL_FRAUD_TYPES = [
    FRAUD_REVENUE_INFLATION,
    FRAUD_EXPENSE_HIDING,
    FRAUD_CHANNEL_STUFFING,
    FRAUD_EARNINGS_SMOOTHING,
    FRAUD_ASSET_OVERSTATEMENT,
]


# ─────────────────────────────────────────────────────────────
# Individual Fraud Injectors
# ─────────────────────────────────────────────────────────────

def inject_revenue_inflation(statements: FinancialStatements, quarters: list[int] = None) -> FinancialStatements:
    """
    FRAUD: Revenue Inflation
    Fake sales are recorded — revenue jumps significantly
    but operating cash flow stays flat (cash didn't actually come in).

    Red flag: revenue spikes while operating cash flow stays the same.

    Args:
        quarters: which quarters to inflate (default: last 2)
    """
    if quarters is None:
        quarters = [2, 3]  # Q3, Q4 by default

    inflation_factor = random.uniform(1.4, 1.8)  # 40-80% inflation

    income = statements.income_statement
    for q in quarters:
        original_revenue = income.revenue[q]
        inflated_revenue = round(original_revenue * inflation_factor, 1)

        # Inflate revenue and derived metrics
        income.revenue[q]       = inflated_revenue
        income.gross_profit[q]  = round(inflated_revenue - income.cost_of_goods_sold[q], 1)
        income.net_income[q]    = round(income.gross_profit[q] - income.operating_expenses[q], 1)

        # Cash flow deliberately NOT updated — this is the fraud signal
        # Real cash didn't come in, so operating cash stays unchanged

    statements.injected_frauds.append({
        "type":        FRAUD_REVENUE_INFLATION,
        "quarters":    quarters,
        "line_item":   "revenue",
        "statement":   "income_statement",
        "signal":      "revenue grew significantly but operating cash flow did not follow",
        "severity":    "high",
    })

    return statements


def inject_expense_hiding(statements: FinancialStatements, quarters: list[int] = None) -> FinancialStatements:
    """
    FRAUD: Expense Hiding
    Operating expenses are artificially reduced by moving costs off the books.
    This inflates net income without any real business improvement.

    Red flag: operating expenses drop suddenly while revenue stays flat.
    """
    if quarters is None:
        quarters = [2, 3]

    reduction_factor = random.uniform(0.4, 0.6)  # expenses cut by 40-60%

    income = statements.income_statement
    for q in quarters:
        original_opex = income.operating_expenses[q]
        reduced_opex  = round(original_opex * reduction_factor, 1)

        income.operating_expenses[q] = reduced_opex
        income.net_income[q] = round(income.gross_profit[q] - reduced_opex, 1)

        # Liabilities should have gone up (costs were deferred, not eliminated)
        # but fraudulently they are not recorded — this is a secondary signal
        # (agent can check: if opex drops but liabilities don't rise, suspicious)

    statements.injected_frauds.append({
        "type":      FRAUD_EXPENSE_HIDING,
        "quarters":  quarters,
        "line_item": "operating_expenses",
        "statement": "income_statement",
        "signal":    "operating expenses dropped sharply with no business explanation",
        "severity":  "high",
    })

    return statements


def inject_channel_stuffing(statements: FinancialStatements, quarters: list[int] = None) -> FinancialStatements:
    """
    FRAUD: Channel Stuffing
    Company ships excess goods to distributors to record revenue early.
    Revenue and receivables both inflate — but cash doesn't come in yet
    (distributors haven't paid and may return the goods).

    Red flag: receivables grow MUCH faster than revenue.
    Normal ratio: receivables ≈ 8-20% of revenue depending on industry.
    Stuffed ratio: receivables balloon to 40-60% of revenue.
    """
    if quarters is None:
        quarters = [2, 3]

    stuffing_factor = random.uniform(2.5, 3.5)  # receivables 2.5-3.5x normal

    income  = statements.income_statement
    balance = statements.balance_sheet

    for q in quarters:
        # Revenue is inflated (goods shipped = revenue recognized)
        original_revenue = income.revenue[q]
        income.revenue[q]       = round(original_revenue * random.uniform(1.2, 1.4), 1)
        income.gross_profit[q]  = round(income.revenue[q] - income.cost_of_goods_sold[q], 1)
        income.net_income[q]    = round(income.gross_profit[q] - income.operating_expenses[q], 1)

        # Receivables balloon — customers haven't paid
        balance.receivables[q] = round(balance.receivables[q] * stuffing_factor, 1)

        # Cash flow stays flat — no actual cash received
        # (operating cash flow unchanged — the fraud signal)

    statements.injected_frauds.append({
        "type":      FRAUD_CHANNEL_STUFFING,
        "quarters":  quarters,
        "line_item": "receivables",
        "statement": "balance_sheet",
        "signal":    "receivables grew far faster than revenue — goods shipped but not paid for",
        "severity":  "high",
    })

    return statements


def inject_earnings_smoothing(statements: FinancialStatements) -> FinancialStatements:
    """
    FRAUD: Earnings Smoothing
    Company manipulates reported earnings to show unnaturally stable growth.
    In good quarters, some profit is hidden in reserves.
    In bad quarters, those reserves are released to meet targets.

    Red flag: earnings growth is suspiciously smooth — real businesses
    have variance. Perfect linear growth quarter after quarter is unnatural.

    This is the SUBTLE fraud used in Task 3.
    """
    income = statements.income_statement

    # Calculate what "natural" net income would be
    avg_income = sum(income.net_income) / 4

    # Replace with artificially smooth version — small, perfect increments
    smooth_increment = avg_income * random.uniform(0.02, 0.04)  # exactly 2-4% each quarter
    smoothed = [round(avg_income + smooth_increment * (i - 1.5), 1) for i in range(4)]

    income.net_income = smoothed

    # Gross profit adjusted to be consistent with smoothed net income
    for i in range(4):
        income.gross_profit[i] = round(smoothed[i] + income.operating_expenses[i], 1)
        income.revenue[i]      = round(income.gross_profit[i] + income.cost_of_goods_sold[i], 1)

    statements.injected_frauds.append({
        "type":      FRAUD_EARNINGS_SMOOTHING,
        "quarters":  [0, 1, 2, 3],
        "line_item": "net_income",
        "statement": "income_statement",
        "signal":    "earnings grew at an unnaturally consistent rate across all 4 quarters",
        "severity":  "medium",
    })

    return statements


def inject_asset_overstatement(statements: FinancialStatements, quarters: list[int] = None) -> FinancialStatements:
    """
    FRAUD: Asset Overstatement
    Total assets are inflated on the balance sheet.
    This makes the company look more valuable than it is.

    Red flag: total assets grow much faster than revenue or net income.
    Assets should roughly track business activity — if assets jump 40%
    but revenue only grew 5%, that's suspicious.
    """
    if quarters is None:
        quarters = [2, 3]

    inflation_factor = random.uniform(1.3, 1.5)

    balance = statements.balance_sheet
    for q in quarters:
        balance.total_assets[q] = round(balance.total_assets[q] * inflation_factor, 1)
        # Equity inflated to balance the sheet (fraudulent)
        balance.equity[q] = round(balance.total_assets[q] - balance.total_liabilities[q], 1)

    statements.injected_frauds.append({
        "type":      FRAUD_ASSET_OVERSTATEMENT,
        "quarters":  quarters,
        "line_item": "total_assets",
        "statement": "balance_sheet",
        "signal":    "total assets grew disproportionately compared to revenue and business activity",
        "severity":  "medium",
    })

    return statements


# ─────────────────────────────────────────────────────────────
# Task-Level Injectors
# These combine individual frauds for each difficulty level.
# ─────────────────────────────────────────────────────────────

def inject_easy(statements: FinancialStatements, seed: int = None) -> FinancialStatements:
    """
    Task 1 — Easy
    One obvious fraud: revenue inflation.
    Revenue jumps 40-80% in Q3-Q4 but cash flow stays flat.
    A competent LLM should catch this.
    """
    if seed is not None:
        random.seed(seed)

    return inject_revenue_inflation(statements, quarters=[2, 3])


def inject_medium(statements: FinancialStatements, seed: int = None) -> FinancialStatements:
    """
    Task 2 — Medium
    Two frauds: revenue inflation + channel stuffing.
    Agent must cross-reference income statement AND balance sheet.
    Numbers look okay in isolation — only suspicious when compared.
    """
    if seed is not None:
        random.seed(seed)

    statements = inject_revenue_inflation(statements, quarters=[2, 3])
    statements = inject_expense_hiding(statements, quarters=[1, 2])
    return statements


def inject_hard(statements: FinancialStatements, seed: int = None) -> FinancialStatements:
    """
    Task 3 — Hard
    Three frauds layered together:
      1. Earnings smoothing (subtle — requires statistical reasoning)
      2. Channel stuffing (requires cross-statement comparison)
      3. Asset overstatement (requires benchmark comparison)

    Red herrings present: some metrics look unusual due to normal
    business variation, not fraud. Agent must distinguish signal from noise.
    Frontier LLMs will genuinely struggle here.
    """
    if seed is not None:
        random.seed(seed)

    statements = inject_earnings_smoothing(statements)
    statements = inject_channel_stuffing(statements, quarters=[1, 2, 3])
    statements = inject_asset_overstatement(statements, quarters=[2, 3])
    return statements


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_generator import generate_clean_statements

    print("=" * 55)
    print("TASK 1 — EASY (revenue inflation)")
    print("=" * 55)
    s = inject_easy(generate_clean_statements(seed=42), seed=1)
    print(f"Revenue:          {s.income_statement.revenue}")
    print(f"Operating CF:     {s.cash_flow.operating}")
    print(f"Injected frauds:  {[f['type'] for f in s.injected_frauds]}")

    print("\n" + "=" * 55)
    print("TASK 2 — MEDIUM (revenue inflation + expense hiding)")
    print("=" * 55)
    s = inject_medium(generate_clean_statements(seed=42), seed=2)
    print(f"Revenue:          {s.income_statement.revenue}")
    print(f"Operating Expenses:{s.income_statement.operating_expenses}")
    print(f"Injected frauds:  {[f['type'] for f in s.injected_frauds]}")

    print("\n" + "=" * 55)
    print("TASK 3 — HARD (3 layered frauds)")
    print("=" * 55)
    s = inject_hard(generate_clean_statements(seed=42), seed=3)
    print(f"Net Income:       {s.income_statement.net_income}")
    print(f"Receivables:      {s.balance_sheet.receivables}")
    print(f"Total Assets:     {s.balance_sheet.total_assets}")
    print(f"Injected frauds:  {[f['type'] for f in s.injected_frauds]}")