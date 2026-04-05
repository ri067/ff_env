# Financial Fraud Detection Environment

An [OpenEnv](https://huggingface.co/openenv)-compatible reinforcement learning environment where an AI agent plays the role of a **forensic accountant** auditing synthetic company financial statements to detect accounting fraud.

---

## Overview

The agent receives 4 quarters of financial data across 3 statements (Income Statement, Balance Sheet, Cash Flow) for a fictitious company. One or more fraud patterns have been secretly injected into the data. The agent must investigate, identify the fraud, and submit a report — all within a step budget.

This simulates a real task that financial auditors perform daily: detecting manipulation in reported financials before it reaches investors or regulators.

---

## Fraud Patterns

| Fraud Type | Description | Key Signal |
|---|---|---|
| `revenue_inflation` | Fake sales recorded to inflate revenue | Revenue spikes but operating cash flow stays flat |
| `expense_hiding` | Operating costs moved off the books | Expenses drop sharply with no business explanation |
| `channel_stuffing` | Excess goods shipped to inflate sales | Receivables grow far faster than revenue |
| `earnings_smoothing` | Profits manipulated to show perfect steady growth | Unnaturally low variance in quarterly earnings |
| `asset_overstatement` | Assets inflated on the balance sheet | Assets grow much faster than revenue |

---

## Tasks

| Task | Difficulty | Frauds | Step Budget | Expected Score (Frontier LLM) |
|---|---|---|---|---|
| `easy` | Easy | 1 — revenue inflation | 15 | ~1.0 |
| `medium` | Medium | 2 — revenue inflation + expense hiding | 20 | ~0.85 |
| `hard` | Hard | 3 — earnings smoothing + channel stuffing + asset overstatement | 25 | ~0.55 |

---

## Action Space

The agent takes one action per step by sending a JSON object:

| Action | Description | Example |
|---|---|---|
| `inspect` | View a metric across all 4 quarters with growth analysis | `{"action_type": "inspect", "parameters": {"statement": "income_statement", "metric": "revenue"}}` |
| `compare` | Cross-reference two metrics to spot divergence | `{"action_type": "compare", "parameters": {"metric_a": "revenue", "metric_b": "operating_cashflow"}}` |
| `flag` | Raise a fraud alert | `{"action_type": "flag", "parameters": {"fraud_type": "revenue_inflation", "line_item": "revenue", "statement": "income_statement", "quarters": [2, 3]}}` |
| `check_benchmark` | Compare a metric against industry average | `{"action_type": "check_benchmark", "parameters": {"metric": "receivables_ratio"}}` |
| `request_detail` | Get full breakdown of one quarter | `{"action_type": "request_detail", "parameters": {"quarter": 2}}` |
| `submit_report` | Finalise findings and end episode | `{"action_type": "submit_report", "parameters": {}}` |

---

## Observation Space

Each step returns a `FraudObservation` containing:

- **Company info** — name, industry, quarters
- **Income Statement** — revenue, COGS, gross profit, operating expenses, net income
- **Balance Sheet** — cash, receivables, inventory, total assets, liabilities, equity
- **Cash Flow Statement** — operating, investing, financing, net change
- **Industry benchmarks** — gross margin, receivables ratio for the company's sector
- **Episode state** — flags raised so far, steps remaining, last action result
- **Task info** — task name and description

---

## Reward Function

Rewards are given throughout the episode (not just at the end):

| Event | Reward |
|---|---|
| Correct fraud type flagged | +0.25 |
| Correct line item identified | +0.10 |
| Relevant cross-statement comparison | +0.10 |
| Inspection / exploration | +0.05 |
| False positive (wrong fraud flagged) | -0.10 |
| Unknown action type | -0.05 |
| Final submit — graded 0.0–1.0 | full episode score |

Final score breakdown (on submit):
- **Fraud detection** — 60% (how many injected frauds found)
- **Precision** — 20% (penalises false positives)
- **Efficiency** — 20% (rewards solving quickly)

---

## Setup

### Requirements
- Python 3.10–3.12
- Docker
- `uv` package manager

### Install

```bash
git clone <your-repo-url>
cd ff_env
uv sync
```

### Run locally

```bash
cd ff_env
uv run server
```

Server starts at `http://localhost:8000`.

### Run with Docker

```bash
cd ff_env/ff_env
docker build -t ff_env:latest -f server/Dockerfile .
docker run -p 8000:8000 ff_env:latest
```

---

## Baseline Inference

Run the LLM baseline against all 3 tasks:

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your-api-key"

python inference.py
```

### Baseline Results (llama-3.3-70b-versatile)

| Task | Score | Steps Used |
|---|---|---|
| Easy | 1.000 | 6 |
| Medium | 1.000 | 9 |
| Hard | 0.573 | 14 |
| **Average** | **0.858** | |

---

## Project Structure

```
ff_env/
├── inference.py               # Baseline inference script (run this)
├── baseline_results.json      # Saved baseline scores
├── ff_env/
│   ├── models.py              # Pydantic Action + Observation models
│   ├── data_generator.py      # Generates synthetic financial statements
│   ├── fraud_injector.py      # Injects fraud patterns into statements
│   ├── grader.py              # Scores agent performance 0.0–1.0
│   ├── client.py              # OpenEnv client
│   ├── openenv.yaml           # OpenEnv spec config
│   └── server/
│       ├── app.py             # FastAPI server
│       ├── ff_env_environment.py  # Main environment (reset/step/state)
│       ├── Dockerfile         # Container definition
│       └── requirements.txt   # Server dependencies
```

---

## Environment Design Notes

**Why synthetic data?**
All financial statements are procedurally generated — no real company data is used. Fraud is injected deterministically using a seed, so episodes are fully reproducible.

**Why these fraud types?**
These are the most common real-world accounting fraud schemes documented in forensic accounting literature (Enron-style revenue inflation, WorldCom-style expense hiding, channel stuffing as seen in Sunbeam, etc.).

**Why is hard genuinely hard?**
The hard task layers 3 fraud types simultaneously, includes red herrings (metrics that look abnormal due to normal business variation), and requires the agent to reason across all 3 financial statements and compare against industry benchmarks. Frontier LLMs score ~0.55 on this task.