"""
models.py

Pydantic models for the Financial Fraud Detection environment.
These define the typed interface that OpenEnv requires:
  - FraudAction    → what the agent can do each step
  - FraudObservation → what the agent sees after each step

OpenEnv requires these to be Pydantic models that extend
Action and Observation from openenv.core.
"""

from typing import Optional
from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ─────────────────────────────────────────────────────────────
# Action
# What the agent can do each step.
# The agent picks ONE action per step.
# ─────────────────────────────────────────────────────────────

class FraudAction(Action):
    """
    One action the agent takes per step.

    The agent communicates via two fields:
      - action_type: which kind of action (see list below)
      - parameters: a dict with the specifics of that action

    Valid action_types and their parameters:

    1. "inspect"
       Look closely at a specific metric in a specific statement.
       parameters: {"statement": "income_statement", "metric": "revenue"}

    2. "compare"
       Cross-reference two metrics to spot inconsistencies.
       parameters: {"metric_a": "revenue", "metric_b": "operating_cashflow"}

    3. "flag"
       Raise a fraud alert — agent believes it found something.
       parameters: {"fraud_type": "revenue_inflation", "line_item": "revenue",
                    "statement": "income_statement", "quarters": [2, 3]}

    4. "check_benchmark"
       Compare a metric against industry average.
       parameters: {"metric": "receivables_ratio"}

    5. "request_detail"
       Ask for a closer look at a specific quarter.
       parameters: {"quarter": 2, "statement": "cash_flow"}

    6. "submit_report"
       Agent finalises its findings and ends the episode.
       parameters: {} (empty — triggers grader)
    """

    action_type: str = Field(
        ...,
        description=(
            "Type of action to take. One of: "
            "'inspect', 'compare', 'flag', 'check_benchmark', "
            "'request_detail', 'submit_report'"
        )
    )

    parameters: dict = Field(
        default_factory=dict,
        description="Parameters for the action. Contents depend on action_type."
    )


# ─────────────────────────────────────────────────────────────
# Observation
# What the agent sees after each step (or after reset).
# ─────────────────────────────────────────────────────────────

class FraudObservation(Observation):
    """
    What the agent sees at each step.

    Always includes the full financial statements so the agent
    can reason about them freely.

    Also includes:
      - flags_raised: what the agent has flagged so far
      - last_action_result: feedback on what just happened
      - step_budget: how many steps remain
      - task_description: natural language description of the task
    """

    # ── Company Info ──────────────────────────────────────────
    company_name: str = Field(default="", description="Name of the company being audited")
    industry:     str = Field(default="", description="Industry sector of the company")
    quarters:     list[str] = Field(default_factory=list, description="Quarter labels e.g. ['Q1 2023', ...]")

    # ── Income Statement ──────────────────────────────────────
    revenue:             list[float] = Field(default_factory=list, description="Quarterly revenue ($K)")
    cost_of_goods_sold:  list[float] = Field(default_factory=list, description="Quarterly COGS ($K)")
    gross_profit:        list[float] = Field(default_factory=list, description="Quarterly gross profit ($K)")
    operating_expenses:  list[float] = Field(default_factory=list, description="Quarterly operating expenses ($K)")
    net_income:          list[float] = Field(default_factory=list, description="Quarterly net income ($K)")

    # ── Balance Sheet ─────────────────────────────────────────
    cash:               list[float] = Field(default_factory=list, description="Quarterly cash on hand ($K)")
    receivables:        list[float] = Field(default_factory=list, description="Quarterly accounts receivable ($K)")
    inventory:          list[float] = Field(default_factory=list, description="Quarterly inventory ($K)")
    total_assets:       list[float] = Field(default_factory=list, description="Quarterly total assets ($K)")
    total_liabilities:  list[float] = Field(default_factory=list, description="Quarterly total liabilities ($K)")
    equity:             list[float] = Field(default_factory=list, description="Quarterly equity ($K)")

    # ── Cash Flow Statement ───────────────────────────────────
    operating_cashflow:  list[float] = Field(default_factory=list, description="Quarterly operating cash flow ($K)")
    investing_cashflow:  list[float] = Field(default_factory=list, description="Quarterly investing cash flow ($K)")
    financing_cashflow:  list[float] = Field(default_factory=list, description="Quarterly financing cash flow ($K)")
    net_cash_change:     list[float] = Field(default_factory=list, description="Quarterly net cash change ($K)")

    # ── Industry Benchmarks (always visible to agent) ─────────
    benchmark_gross_margin:     float = Field(default=0.0, description="Industry average gross margin ratio")
    benchmark_receivables_ratio: float = Field(default=0.0, description="Industry average receivables/revenue ratio")

    # ── Episode State ─────────────────────────────────────────
    flags_raised: list[dict] = Field(
        default_factory=list,
        description="Fraud flags the agent has raised so far this episode"
    )
    step_budget: int = Field(
        default=20,
        description="Number of steps remaining before episode ends"
    )
    last_action_result: str = Field(
        default="",
        description="Feedback on the last action taken (what happened)"
    )

    # ── Task Info ─────────────────────────────────────────────
    task_name: str = Field(
        default="",
        description="Name of the current task (easy / medium / hard)"
    )
    task_description: str = Field(
        default="",
        description="Natural language description of what the agent should do"
    )