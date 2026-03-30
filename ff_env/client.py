"""Ff Env Environment Client."""
from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from .models import FraudAction, FraudObservation


class FfEnv(EnvClient[FraudAction, FraudObservation, State]):
    """
    Client for the Financial Fraud Detection Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> with FfEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(FraudAction(
        ...         action_type="inspect",
        ...         parameters={"statement": "income_statement", "metric": "revenue"}
        ...     ))
        ...     print(result.observation.last_action_result)
    """

    def _step_payload(self, action: FraudAction) -> Dict:
        return {
            "action_type": action.action_type,
            "parameters":  action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[FraudObservation]:
        obs_data = payload.get("observation", {})
        observation = FraudObservation(
            # Company info
            company_name=obs_data.get("company_name", ""),
            industry=obs_data.get("industry", ""),
            quarters=obs_data.get("quarters", []),
            # Income statement
            revenue=obs_data.get("revenue", []),
            cost_of_goods_sold=obs_data.get("cost_of_goods_sold", []),
            gross_profit=obs_data.get("gross_profit", []),
            operating_expenses=obs_data.get("operating_expenses", []),
            net_income=obs_data.get("net_income", []),
            # Balance sheet
            cash=obs_data.get("cash", []),
            receivables=obs_data.get("receivables", []),
            inventory=obs_data.get("inventory", []),
            total_assets=obs_data.get("total_assets", []),
            total_liabilities=obs_data.get("total_liabilities", []),
            equity=obs_data.get("equity", []),
            # Cash flow
            operating_cashflow=obs_data.get("operating_cashflow", []),
            investing_cashflow=obs_data.get("investing_cashflow", []),
            financing_cashflow=obs_data.get("financing_cashflow", []),
            net_cash_change=obs_data.get("net_cash_change", []),
            # Benchmarks
            benchmark_gross_margin=obs_data.get("benchmark_gross_margin", 0.0),
            benchmark_receivables_ratio=obs_data.get("benchmark_receivables_ratio", 0.0),
            # Episode state
            flags_raised=obs_data.get("flags_raised", []),
            step_budget=obs_data.get("step_budget", 0),
            last_action_result=obs_data.get("last_action_result", ""),
            # Task info
            task_name=obs_data.get("task_name", ""),
            task_description=obs_data.get("task_description", ""),
            # OpenEnv fields
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )