"""
data_generator.py

Generates clean (fraud-free) synthetic financial statements for a fake company.
Produces 4 quarters of data across 3 statements:
  - Income Statement
  - Balance Sheet
  - Cash Flow Statement

All numbers are internally consistent and realistic.
The fraud_injector.py will later deliberately manipulate specific values.
"""

import random
from dataclasses import dataclass, field


# ─────────────────────────────────────────────────────────────
# Data Structures
# Plain Python dataclasses — no Pydantic yet.
# Pydantic comes in models.py for the OpenEnv API layer.
# ─────────────────────────────────────────────────────────────

@dataclass
class IncomeStatement:
    """
    What the company earned and spent each quarter.
    All values in $thousands.

    Key relationship:
        gross_profit = revenue - cost_of_goods_sold
        net_income   = gross_profit - operating_expenses
    """
    revenue:             list[float]
    cost_of_goods_sold:  list[float]
    gross_profit:        list[float]
    operating_expenses:  list[float]
    net_income:          list[float]


@dataclass
class BalanceSheet:
    """
    Snapshot of what the company owns and owes at end of each quarter.
    All values in $thousands.

    Key relationship:
        equity = total_assets - total_liabilities
    """
    cash:               list[float]   # liquid cash on hand
    receivables:        list[float]   # money customers owe the company
    inventory:          list[float]   # goods in stock
    total_assets:       list[float]   # everything the company owns
    total_liabilities:  list[float]   # everything the company owes
    equity:             list[float]   # owner's stake


@dataclass
class CashFlowStatement:
    """
    Actual cash moving in and out each quarter.
    All values in $thousands.

    THE most important fraud signal:
        operating cash flow should roughly match net income.
        If profits are high but cash is flat → suspicious.
    """
    operating:   list[float]   # cash from normal business
    investing:   list[float]   # cash spent on equipment etc (usually negative)
    financing:   list[float]   # cash from loans or stock issuance
    net_change:  list[float]   # operating + investing + financing


@dataclass
class FinancialStatements:
    """
    Full package — all 3 statements for one company over 4 quarters.
    This is what the agent receives as its observation each episode.
    """
    company_name:       str
    quarters:           list[str]   # ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"]
    industry:           str

    income_statement:   IncomeStatement   = field(default_factory=lambda: None)
    balance_sheet:      BalanceSheet      = field(default_factory=lambda: None)
    cash_flow:          CashFlowStatement = field(default_factory=lambda: None)

    # Ground truth — populated by fraud_injector.py, read by grader.py
    # The agent NEVER sees this directly. It must infer from the numbers.
    injected_frauds: list[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

COMPANY_NAMES = [
    "Acme Corp", "Zenith Industries", "Pinnacle Holdings",
    "Atlas Manufacturing", "Crestwood Enterprises", "Meridian Group",
    "Vortex Technologies", "Apex Solutions", "Summit Enterprises"
]

INDUSTRIES = ["Manufacturing", "Retail", "Technology", "Healthcare", "Energy"]

# Normal financial ratios per industry.
# Used by the agent in Task 3 to compare against benchmarks.
INDUSTRY_BENCHMARKS = {
    "Manufacturing": {"gross_margin": 0.35, "receivables_ratio": 0.15, "liab_ratio": 0.45},
    "Retail":        {"gross_margin": 0.25, "receivables_ratio": 0.08, "liab_ratio": 0.50},
    "Technology":    {"gross_margin": 0.65, "receivables_ratio": 0.20, "liab_ratio": 0.35},
    "Healthcare":    {"gross_margin": 0.50, "receivables_ratio": 0.18, "liab_ratio": 0.40},
    "Energy":        {"gross_margin": 0.30, "receivables_ratio": 0.12, "liab_ratio": 0.48},
}


# ─────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────

def generate_clean_statements(seed: int = None) -> FinancialStatements:
    """
    Generates a complete, internally consistent set of financial statements.
    All 3 statements are consistent with each other — no fraud injected yet.

    Args:
        seed: random seed for reproducibility (grader needs determinism)

    Returns:
        FinancialStatements with injected_frauds=[] (clean)
    """
    if seed is not None:
        random.seed(seed)

    company  = random.choice(COMPANY_NAMES)
    industry = random.choice(INDUSTRIES)
    quarters = ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"]
    bench    = INDUSTRY_BENCHMARKS[industry]

    # ── Income Statement ──────────────────────────────────────

    # Base revenue with mild organic growth (2–6% per quarter)
    base_revenue = random.uniform(800, 2000)
    growth_rate  = random.uniform(0.02, 0.06)
    revenue = [
        round(base_revenue * (1 + growth_rate) ** q, 1)
        for q in range(4)
    ]

    # COGS derived from industry gross margin
    gross_margin = bench["gross_margin"] + random.uniform(-0.03, 0.03)
    cogs         = [round(r * (1 - gross_margin), 1) for r in revenue]
    gross_profit = [round(revenue[i] - cogs[i], 1) for i in range(4)]

    # Operating expenses: 15–20% of revenue
    opex_ratio         = random.uniform(0.15, 0.20)
    operating_expenses = [round(r * opex_ratio, 1) for r in revenue]
    net_income         = [round(gross_profit[i] - operating_expenses[i], 1) for i in range(4)]

    income = IncomeStatement(
        revenue=revenue,
        cost_of_goods_sold=cogs,
        gross_profit=gross_profit,
        operating_expenses=operating_expenses,
        net_income=net_income,
    )

    # ── Balance Sheet ─────────────────────────────────────────

    base_assets  = random.uniform(3000, 8000)
    asset_growth = random.uniform(0.01, 0.03)
    total_assets = [round(base_assets * (1 + asset_growth) ** q, 1) for q in range(4)]

    # Receivables: industry-specific % of revenue
    recv_ratio   = bench["receivables_ratio"] + random.uniform(-0.02, 0.02)
    receivables  = [round(r * recv_ratio, 1) for r in revenue]

    # Cash grows with accumulated net income
    cash = [
        round(total_assets[0] * 0.10 + sum(net_income[:i+1]) * 0.5, 1)
        for i in range(4)
    ]

    inventory         = [round(a * 0.08, 1) for a in total_assets]
    liab_ratio        = bench["liab_ratio"] + random.uniform(-0.03, 0.03)
    total_liabilities = [round(a * liab_ratio, 1) for a in total_assets]
    equity            = [round(total_assets[i] - total_liabilities[i], 1) for i in range(4)]

    balance = BalanceSheet(
        cash=cash,
        receivables=receivables,
        inventory=inventory,
        total_assets=total_assets,
        total_liabilities=total_liabilities,
        equity=equity,
    )

    # ── Cash Flow Statement ───────────────────────────────────

    # Operating cash flow ≈ net income (small working capital noise)
    # This is the KEY consistency check — fraud breaks this relationship
    operating = [round(net_income[i] * random.uniform(0.90, 1.10), 1) for i in range(4)]
    investing  = [round(random.uniform(-150, -50), 1) for _ in range(4)]
    financing  = [round(random.uniform(-30, 50), 1)  for _ in range(4)]
    net_change = [round(operating[i] + investing[i] + financing[i], 1) for i in range(4)]

    cashflow = CashFlowStatement(
        operating=operating,
        investing=investing,
        financing=financing,
        net_change=net_change,
    )

    return FinancialStatements(
        company_name=company,
        quarters=quarters,
        industry=industry,
        income_statement=income,
        balance_sheet=balance,
        cash_flow=cashflow,
        injected_frauds=[],
    )


# ─────────────────────────────────────────────────────────────
# Quick test — run this file directly to verify output
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    s = generate_clean_statements(seed=42)

    print(f"Company : {s.company_name}  ({s.industry})")
    print(f"Quarters: {s.quarters}\n")

    print("── Income Statement ($K) ──────────────────────────")
    print(f"  Revenue          : {s.income_statement.revenue}")
    print(f"  COGS             : {s.income_statement.cost_of_goods_sold}")
    print(f"  Gross Profit     : {s.income_statement.gross_profit}")
    print(f"  Operating Expenses: {s.income_statement.operating_expenses}")
    print(f"  Net Income       : {s.income_statement.net_income}")

    print("\n── Balance Sheet ($K) ─────────────────────────────")
    print(f"  Cash             : {s.balance_sheet.cash}")
    print(f"  Receivables      : {s.balance_sheet.receivables}")
    print(f"  Inventory        : {s.balance_sheet.inventory}")
    print(f"  Total Assets     : {s.balance_sheet.total_assets}")
    print(f"  Total Liabilities: {s.balance_sheet.total_liabilities}")
    print(f"  Equity           : {s.balance_sheet.equity}")

    print("\n── Cash Flow Statement ($K) ───────────────────────")
    print(f"  Operating        : {s.cash_flow.operating}")
    print(f"  Investing        : {s.cash_flow.investing}")
    print(f"  Financing        : {s.cash_flow.financing}")
    print(f"  Net Change       : {s.cash_flow.net_change}")

    print("\n── Fraud Check ────────────────────────────────────")
    print(f"  Injected frauds  : {s.injected_frauds}  ← should be empty")