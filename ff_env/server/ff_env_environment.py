"""
ff_env_environment.py

The Financial Fraud Detection Environment.

An AI agent plays the role of a forensic accountant auditing a company's
financial statements. It must identify fraud patterns by inspecting,
comparing, and flagging suspicious numbers across 3 financial statements.

Episode flow:
    reset() -> agent receives financial statements (with hidden fraud)
    step(action) -> agent inspects/flags -> receives reward + feedback
    step(submit_report) -> episode ends -> grader scores final flags
"""

from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FraudAction, FraudObservation
except ImportError:
    from models import FraudAction, FraudObservation

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data_generator import generate_clean_statements, INDUSTRY_BENCHMARKS
except ImportError:
    from ff_env.data_generator import generate_clean_statements, INDUSTRY_BENCHMARKS
try:
    from fraud_injector import inject_easy, inject_medium, inject_hard
except ImportError:
    from ff_env.fraud_injector import inject_easy, inject_medium, inject_hard
try:
    from grader import grade_episode
except ImportError:
    from ff_env.grader import grade_episode



#Task Definitions

TASKS = {
    "easy": {
        "name": "easy",
        "injector": inject_easy,
        "step_budget": 15,
        "description": (
            "Audit the financial statements of a company over 4 quarters. "
            "One fraud has been committed. "
            "Inspect the statements, identify the suspicious pattern, "
            "flag it with the correct fraud type and line item, "
            "then submit your report. "
            "Hint: compare revenue growth against cash flow trends."
        ),
        "num_frauds": 1,
    },
    "medium": {
        "name": "medium",
        "injector": inject_medium,
        "step_budget": 20,
        "description": (
            "Audit the financial statements of a company over 4 quarters. "
            "Two frauds have been committed across different statements. "
            "You must find both. Cross-reference the income statement "
            "and balance sheet carefully. "
            "Submit your report only when you have flagged all issues."
        ),
        "num_frauds": 2,
    },
    "hard": {
        "name": "hard",
        "injector": inject_hard,
        "step_budget": 25,
        "description": (
            "Audit the financial statements of a company over 4 quarters. "
            "Three sophisticated frauds have been layered together. "
            "Some anomalies are red herrings — normal business variation. "
            "You must distinguish real fraud signals from noise. "
            "Use industry benchmarks, cross-statement comparisons, and "
            "statistical reasoning. All three frauds must be identified."
        ),
        "num_frauds": 3,
    },
}

VALID_ACTION_TYPES = {
    "inspect", "compare", "flag",
    "check_benchmark", "request_detail", "submit_report"
}

VALID_FRAUD_TYPES = {
    "revenue_inflation", "expense_hiding", "channel_stuffing",
    "earnings_smoothing", "asset_overstatement"
}


#Environment
class FfEnvironment(Environment):
    """
    Financial Fraud Detection Environment.

    The agent audits synthetic financial statements and must identify
    injected fraud patterns by taking investigative actions.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._statements = None
        self._flags_raised = []
        self._steps_remaining = 0
        self._task = TASKS["easy"]
        self._episode_seed = 0
        self._done = False

    #reset()
    def reset(self, task_name: str = "easy", seed: int = None) -> FraudObservation:
        """
        Start a fresh episode.

        Generates clean financial statements, injects fraud,
        and returns the initial observation.

        Args:
            task_name: "easy", "medium", or "hard"
            seed: for reproducibility
        """
        import random

        #Pick task
        task_name = task_name if task_name in TASKS else "easy"
        self._task = TASKS[task_name]

        #Generate seed
        self._episode_seed = seed if seed is not None else random.randint(0, 99999)

        #Generate clean statements then inject fraud
        statements = generate_clean_statements(seed=self._episode_seed)
        self._statements = self._task["injector"](statements, seed=self._episode_seed + 1)

        #Reset episode state
        self._flags_raised = []
        self._steps_remaining = self._task["step_budget"]
        self._done = False
        self._state = State(episode_id=str(uuid4()), step_count=0)

        return self._build_observation(
            last_action_result=(
                f"New episode started. Task: {task_name.upper()}. "
                f"You have {self._steps_remaining} steps. "
                "Audit the financial statements and flag any fraud you find."
            )
        )

    #step()
    def step(self, action: FraudAction) -> FraudObservation:
        """
        Execute one agent action.

        Returns updated observation with reward and feedback.
        """
        if self._done:
            return self._build_observation(
                last_action_result="Episode is already over. Call reset() to start a new one.",
                reward=0.0,
                done=True,
            )

        self._state.step_count += 1
        self._steps_remaining -= 1

        #Route to the correct action handler
        action_type = action.action_type.lower().strip()
        params      = action.parameters or {}

        if action_type not in VALID_ACTION_TYPES:
            result = (
                f"Unknown action '{action_type}'. "
                f"Valid actions: {sorted(VALID_ACTION_TYPES)}"
            )
            reward = -0.05
        elif action_type == "inspect":
            result, reward = self._handle_inspect(params)
        elif action_type == "compare":
            result, reward = self._handle_compare(params)
        elif action_type == "flag":
            result, reward = self._handle_flag(params)
        elif action_type == "check_benchmark":
            result, reward = self._handle_benchmark(params)
        elif action_type == "request_detail":
            result, reward = self._handle_detail(params)
        elif action_type == "submit_report":
            result, reward = self._handle_submit()

        # Force end if out of steps
        if self._steps_remaining <= 0 and not self._done:
            self._done = True
            result += " [Step budget exhausted — episode ended.]"

        return self._build_observation(
            last_action_result=result,
            reward=reward,
            done=self._done,
        )

    #Action Handlers
    def _handle_inspect(self, params: dict) -> tuple[str, float]:
        """
        Agent inspects a specific metric in a statement.
        Returns the values across all 4 quarters with basic analysis.
        """
        statement = params.get("statement", "")
        metric    = params.get("metric", "")
        s         = self._statements

        data = self._get_metric(statement, metric)
        if data is None:
            return f"Unknown statement or metric: '{statement}.{metric}'", -0.05

        # Basic analysis: compute quarter-over-quarter growth
        growth = []
        for i in range(1, 4):
            if data[i-1] != 0:
                pct = round((data[i] - data[i-1]) / abs(data[i-1]) * 100, 1)
                growth.append(f"Q{i+1}: {'+' if pct >= 0 else ''}{pct}%")

        result = (
            f"[INSPECT] {statement}.{metric}\n"
            f"  Values : {data}\n"
            f"  QoQ growth: {', '.join(growth)}\n"
            f"  Quarters: {s.quarters}"
        )
        return result, 0.05  # small positive reward for exploration

    def _handle_compare(self, params: dict) -> tuple[str, float]:
        """
        Agent cross-references two metrics.
        Returns both series and highlights divergences.
        """
        metric_a = params.get("metric_a", "")
        metric_b = params.get("metric_b", "")

        # Allow shorthand names
        stmt_a, key_a = self._resolve_metric_name(metric_a)
        stmt_b, key_b = self._resolve_metric_name(metric_b)

        data_a = self._get_metric(stmt_a, key_a)
        data_b = self._get_metric(stmt_b, key_b)

        if data_a is None or data_b is None:
            return f"Could not find metrics: '{metric_a}' or '{metric_b}'", -0.05

        # Compute ratios to highlight divergence
        ratios = []
        for i in range(4):
            if data_b[i] != 0:
                r = round(data_a[i] / data_b[i], 2)
                ratios.append(r)

        result = (
            f"[COMPARE] {metric_a} vs {metric_b}\n"
            f"  {metric_a}: {data_a}\n"
            f"  {metric_b}: {data_b}\n"
            f"  Ratio ({metric_a}/{metric_b}): {ratios}"
        )

        # Bonus reward if this comparison is relevant to an injected fraud
        reward = 0.05
        if self._compare_is_relevant(metric_a, metric_b):
            reward = 0.10

        return result, reward

    def _handle_flag(self, params: dict) -> tuple[str, float]:
        """
        Agent raises a fraud flag.
        Checked against injected_frauds for correctness.
        """
        fraud_type = params.get("fraud_type", "").lower().strip()
        line_item  = params.get("line_item", "").lower().strip()
        statement  = params.get("statement", "").lower().strip()
        quarters   = params.get("quarters", [])

        if fraud_type not in VALID_FRAUD_TYPES:
            return (
                f"Unknown fraud type '{fraud_type}'. "
                f"Valid types: {sorted(VALID_FRAUD_TYPES)}",
                -0.05
            )

        # Check if already flagged this fraud type
        already_flagged = any(
            f["fraud_type"] == fraud_type for f in self._flags_raised
        )
        if already_flagged:
            return f"You already flagged '{fraud_type}' — no double counting.", -0.05

        # Check against ground truth
        injected_types = [f["type"] for f in self._statements.injected_frauds]

        if fraud_type in injected_types:
            # Correct fraud type identified
            injected = next(f for f in self._statements.injected_frauds if f["type"] == fraud_type)
            reward = 0.25

            # Bonus for correct line item
            if line_item == injected.get("line_item", ""):
                reward += 0.10

            self._flags_raised.append({
                "fraud_type": fraud_type,
                "line_item":  line_item,
                "statement":  statement,
                "quarters":   quarters,
                "correct":    True,
            })
            result = (
                f"[FLAG RAISED] '{fraud_type}' on {statement}.{line_item}\n"
                f"  This looks suspicious. Noted in your report."
            )
        else:
            # False positive
            reward = -0.10
            self._flags_raised.append({
                "fraud_type": fraud_type,
                "line_item":  line_item,
                "statement":  statement,
                "quarters":   quarters,
                "correct":    False,
            })
            result = (
                f"[FLAG RAISED] '{fraud_type}' on {statement}.{line_item}\n"
                f"  Noted in your report. (Be careful — false positives hurt your score.)"
            )

        return result, reward

    def _handle_benchmark(self, params: dict) -> tuple[str, float]:
        """
        Agent compares a metric against industry average.
        Useful for detecting asset overstatement and channel stuffing.
        """
        metric   = params.get("metric", "").lower().strip()
        s        = self._statements
        bench    = INDUSTRY_BENCHMARKS[s.industry]

        if metric == "receivables_ratio":
            actual = [
                round(s.balance_sheet.receivables[i] / s.income_statement.revenue[i], 3)
                for i in range(4)
            ]
            expected = bench["receivables_ratio"]
            result = (
                f"[BENCHMARK] Receivables/Revenue ratio\n"
                f"  Company : {actual}\n"
                f"  Industry avg ({s.industry}): {expected}\n"
                f"  {'⚠ ABOVE NORMAL' if max(actual) > expected * 1.5 else 'Within normal range'}"
            )
            reward = 0.05 if max(actual) > expected * 1.5 else 0.02

        elif metric == "gross_margin":
            actual = [
                round(s.income_statement.gross_profit[i] / s.income_statement.revenue[i], 3)
                for i in range(4)
            ]
            expected = bench["gross_margin"]
            result = (
                f"[BENCHMARK] Gross Margin\n"
                f"  Company : {actual}\n"
                f"  Industry avg ({s.industry}): {expected}\n"
                f"  {'Within normal range' if abs(sum(actual)/4 - expected) < 0.05 else '⚠ DEVIATES FROM NORM'}"
            )
            reward = 0.05

        elif metric == "asset_growth":
            asset_growth = [
                round((s.balance_sheet.total_assets[i] - s.balance_sheet.total_assets[i-1])
                      / s.balance_sheet.total_assets[i-1] * 100, 1)
                for i in range(1, 4)
            ]
            rev_growth = [
                round((s.income_statement.revenue[i] - s.income_statement.revenue[i-1])
                      / s.income_statement.revenue[i-1] * 100, 1)
                for i in range(1, 4)
            ]
            result = (
                f"[BENCHMARK] Asset Growth vs Revenue Growth\n"
                f"  Asset growth QoQ  : {asset_growth}%\n"
                f"  Revenue growth QoQ: {rev_growth}%\n"
                f"  {'⚠ ASSETS GROWING MUCH FASTER THAN REVENUE' if max(asset_growth) > max(rev_growth) * 2 else 'Proportionate'}"
            )
            reward = 0.05

        else:
            result = (
                f"Unknown benchmark metric '{metric}'. "
                f"Valid options: 'receivables_ratio', 'gross_margin', 'asset_growth'"
            )
            reward = -0.02

        return result, reward

    def _handle_detail(self, params: dict) -> tuple[str, float]:
        """
        Agent requests a detailed breakdown of one quarter.
        Returns all metrics for that quarter side by side.
        """
        quarter = params.get("quarter", 0)
        if not isinstance(quarter, int) or quarter not in range(4):
            return "Invalid quarter. Use 0 (Q1), 1 (Q2), 2 (Q3), or 3 (Q4).", -0.02

        s = self._statements
        q = quarter
        result = (
            f"[DETAIL] {s.quarters[q]}\n"
            f"  Revenue          : {s.income_statement.revenue[q]}\n"
            f"  COGS             : {s.income_statement.cost_of_goods_sold[q]}\n"
            f"  Gross Profit     : {s.income_statement.gross_profit[q]}\n"
            f"  Operating Expenses: {s.income_statement.operating_expenses[q]}\n"
            f"  Net Income       : {s.income_statement.net_income[q]}\n"
            f"  ---\n"
            f"  Total Assets     : {s.balance_sheet.total_assets[q]}\n"
            f"  Receivables      : {s.balance_sheet.receivables[q]}\n"
            f"  Equity           : {s.balance_sheet.equity[q]}\n"
            f"  ---\n"
            f"  Operating CF     : {s.cash_flow.operating[q]}\n"
            f"  Net Cash Change  : {s.cash_flow.net_change[q]}"
        )
        return result, 0.03

    def _handle_submit(self) -> tuple[str, float]:
        """
        Agent submits its final report.
        Triggers the grader and ends the episode.
        """
        self._done = True

        score, breakdown = grade_episode(
            injected_frauds=self._statements.injected_frauds,
            flags_raised=self._flags_raised,
            steps_used=self._state.step_count,
            step_budget=self._task["step_budget"],
        )

        result = (
            f"[REPORT SUBMITTED]\n"
            f"  Final score  : {score:.3f} / 1.000\n"
            f"  Breakdown    : {breakdown}"
        )
        score = min(0.999, max(0.001, score))
        return result, score

    
    
    #state property

    @property
    def state(self) -> State:
        return self._state

    #Helpers
    def _build_observation(
        self,
        last_action_result: str = "",
        reward: float = 0.0,
        done: bool = False,
    ) -> FraudObservation:
        """Build a FraudObservation from current environment state."""
        s     = self._statements
        bench = INDUSTRY_BENCHMARKS.get(s.industry, {}) if s else {}

        return FraudObservation(
            # Company info
            company_name=s.company_name if s else "",
            industry=s.industry if s else "",
            quarters=s.quarters if s else [],

            # Income statement
            revenue=s.income_statement.revenue if s else [],
            cost_of_goods_sold=s.income_statement.cost_of_goods_sold if s else [],
            gross_profit=s.income_statement.gross_profit if s else [],
            operating_expenses=s.income_statement.operating_expenses if s else [],
            net_income=s.income_statement.net_income if s else [],

            # Balance sheet
            cash=s.balance_sheet.cash if s else [],
            receivables=s.balance_sheet.receivables if s else [],
            inventory=s.balance_sheet.inventory if s else [],
            total_assets=s.balance_sheet.total_assets if s else [],
            total_liabilities=s.balance_sheet.total_liabilities if s else [],
            equity=s.balance_sheet.equity if s else [],

            # Cash flow
            operating_cashflow=s.cash_flow.operating if s else [],
            investing_cashflow=s.cash_flow.investing if s else [],
            financing_cashflow=s.cash_flow.financing if s else [],
            net_cash_change=s.cash_flow.net_change if s else [],

            # Benchmarks
            benchmark_gross_margin=bench.get("gross_margin", 0.0),
            benchmark_receivables_ratio=bench.get("receivables_ratio", 0.0),

            # Episode state
            flags_raised=self._flags_raised,
            step_budget=self._steps_remaining,
            last_action_result=last_action_result,

            # Task info
            task_name=self._task["name"] if self._task else "",
            task_description=self._task["description"] if self._task else "",

            # OpenEnv required fields
            reward=reward,
            done=done,
        )

    def _get_metric(self, statement: str, metric: str):
        """Return the list of quarterly values for a given statement.metric."""
        s = self._statements
        mapping = {
            "income_statement": {
                "revenue":            s.income_statement.revenue,
                "cost_of_goods_sold": s.income_statement.cost_of_goods_sold,
                "gross_profit":       s.income_statement.gross_profit,
                "operating_expenses": s.income_statement.operating_expenses,
                "net_income":         s.income_statement.net_income,
            },
            "balance_sheet": {
                "cash":               s.balance_sheet.cash,
                "receivables":        s.balance_sheet.receivables,
                "inventory":          s.balance_sheet.inventory,
                "total_assets":       s.balance_sheet.total_assets,
                "total_liabilities":  s.balance_sheet.total_liabilities,
                "equity":             s.balance_sheet.equity,
            },
            "cash_flow": {
                "operating":          s.cash_flow.operating,
                "investing":          s.cash_flow.investing,
                "financing":          s.cash_flow.financing,
                "net_change":         s.cash_flow.net_change,
            },
        }
        return mapping.get(statement, {}).get(metric, None)

    def _resolve_metric_name(self, name: str) -> tuple[str, str]:
        """Resolve shorthand metric names to (statement, metric) pairs."""
        shortcuts = {
            "revenue":            ("income_statement", "revenue"),
            "net_income":         ("income_statement", "net_income"),
            "operating_expenses": ("income_statement", "operating_expenses"),
            "gross_profit":       ("income_statement", "gross_profit"),
            "receivables":        ("balance_sheet",    "receivables"),
            "total_assets":       ("balance_sheet",    "total_assets"),
            "equity":             ("balance_sheet",    "equity"),
            "operating_cashflow": ("cash_flow",        "operating"),
            "operating":          ("cash_flow",        "operating"),
            "net_change":         ("cash_flow",        "net_change"),
        }
        return shortcuts.get(name.lower(), ("", name))

    def _compare_is_relevant(self, metric_a: str, metric_b: str) -> bool:
        """Returns True if this comparison is diagnostic for any injected fraud."""
        relevant_pairs = {
            frozenset(["revenue", "operating_cashflow"]),
            frozenset(["revenue", "operating"]),
            frozenset(["net_income", "operating_cashflow"]),
            frozenset(["receivables", "revenue"]),
            frozenset(["total_assets", "revenue"]),
            frozenset(["operating_expenses", "revenue"]),
        }
        return frozenset([metric_a.lower(), metric_b.lower()]) in relevant_pairs