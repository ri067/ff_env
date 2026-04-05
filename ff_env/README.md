---
title: Ff Env Environment Server
colorFrom: pink
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - finance
  - fraud-detection
---

# Financial Fraud Detection Environment

An [OpenEnv](https://huggingface.co/openenv)-compatible RL environment where an AI agent plays the role of a **forensic accountant** auditing synthetic company financial statements to detect accounting fraud.

---

## Why This Matters

Financial statement fraud costs investors and markets **over $400 billion annually**. High-profile cases — Enron, WorldCom, Wirecard — have repeatedly shown that even experienced auditors miss sophisticated manipulation schemes hidden across multiple financial documents.

Yet no RL environment exists for training or evaluating agents on this task. Existing OpenEnv environments cover trading, coding, and games — but nothing that requires the **multi-document, cross-statement forensic reasoning** that real auditors perform daily.

This environment fills that gap. It provides a realistic, fully synthetic sandbox where agents can practice detecting the same fraud patterns that forensic accountants are trained to find. A well-trained agent could serve as an automated first-pass auditor — flagging suspicious filings for human review before they reach investors.

---

## Overview

The agent receives 4 quarters of financial data across 3 statements for a fictitious company. One or more fraud patterns have been secretly injected. The agent must investigate by inspecting metrics, comparing statements, checking industry benchmarks, flagging frauds, and submitting a report — all within a step budget.

---

## Fraud Patterns

| Fraud Type | Description | Key Signal |
|---|---|---|
| `revenue_inflation` | Fake sales recorded to inflate revenue | Revenue spikes but cash flow stays flat |
| `expense_hiding` | Operating costs moved off the books | Expenses drop with no explanation |
| `channel_stuffing` | Excess goods shipped to inflate sales | Receivables grow far faster than revenue |
| `earnings_smoothing` | Profits manipulated to show perfect steady growth | Unnaturally low variance in quarterly earnings |
| `asset_overstatement` | Assets inflated on the balance sheet | Assets grow much faster than revenue |

---

## Tasks

| Task | Difficulty | Frauds | Steps | LLM Score |
|---|---|---|---|---|
| `easy` | Easy | 1 — revenue inflation | 15 | ~1.0 |
| `medium` | Medium | 2 — revenue inflation + expense hiding | 20 | ~0.85 |
| `hard` | Hard | 3 — earnings smoothing + channel stuffing + asset overstatement | 25 | ~0.55 |

---

## Action Space

| Action | Description | Example |
|---|---|---|
| `inspect` | View a metric across all 4 quarters | `{"action_type": "inspect", "parameters": {"statement": "income_statement", "metric": "revenue"}}` |
| `compare` | Cross-reference two metrics | `{"action_type": "compare", "parameters": {"metric_a": "revenue", "metric_b": "operating_cashflow"}}` |
| `flag` | Raise a fraud alert | `{"action_type": "flag", "parameters": {"fraud_type": "revenue_inflation", "line_item": "revenue", "statement": "income_statement", "quarters": [2, 3]}}` |
| `check_benchmark` | Compare against industry average | `{"action_type": "check_benchmark", "parameters": {"metric": "receivables_ratio"}}` |
| `request_detail` | Full breakdown of one quarter | `{"action_type": "request_detail", "parameters": {"quarter": 2}}` |
| `submit_report` | Finalise findings, end episode | `{"action_type": "submit_report", "parameters": {}}` |

Valid fraud types: `revenue_inflation`, `expense_hiding`, `channel_stuffing`, `earnings_smoothing`, `asset_overstatement`

---

## Observation Space

Each step returns a `FraudObservation` containing:

- **Company info** — name, industry, quarters
- **Income Statement** — revenue, COGS, gross profit, operating expenses, net income
- **Balance Sheet** — cash, receivables, inventory, total assets, liabilities, equity
- **Cash Flow Statement** — operating, investing, financing, net change
- **Industry benchmarks** — gross margin and receivables ratio for the company's sector
- **Episode state** — flags raised so far, steps remaining, last action result
- **Task info** — task name and natural language description

---

## Reward Function

Rewards are given throughout the episode — not just at the end:

| Event | Reward |
|---|---|
| Correct fraud type flagged | +0.25 |
| Correct line item identified | +0.10 |
| Relevant cross-statement comparison | +0.10 |
| Inspection / exploration | +0.05 |
| False positive (wrong fraud flagged) | -0.10 |
| Unknown action | -0.05 |
| Final submit | graded 0.0–1.0 |

Final score breakdown: detection 60% + precision 20% + efficiency 20%

---

## Quick Start

```bash
# Test live endpoint — no setup needed
curl -X POST https://OrangeUnknown-ff-env.hf.space/reset
```

---

## Run Baseline Inference

```bash
git clone https://huggingface.co/spaces/OrangeUnknown/ff-env
cd ff-env
pip install openenv-core pydantic openai
```

> **API Key Note:** The script reads your LLM key from `OPENAI_API_KEY`.
> Set `API_BASE_URL` and `MODEL_NAME` to match your provider.
> Compatible with any OpenAI-compatible endpoint (OpenAI, Groq, Together, etc.)

```bash
# Using OpenAI
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export OPENAI_API_KEY="your-openai-key"

# Using Groq (free, recommended for testing)
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export OPENAI_API_KEY="your-groq-key"

python3 inference.py
```

---

## Baseline Results (llama-3.3-70b-versatile)

| Task | Score | Steps |
|---|---|---|
| Easy | 1.000 | 6 |
| Medium | 1.000 | 9 |
| Hard | 0.573 | 14 |
| **Average** | **0.858** | |

Fixed seeds ensure the same scenarios every run. Scores saved to `baseline_results.json`.

---

## Local Development

```bash
# Run server locally
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run with Docker
docker build -t ff_env:latest -f Dockerfile .
docker run -p 8000:8000 ff_env:latest
```

---

## Project Structure

```
ff-env/
├── inference.py               # Baseline inference script
├── baseline_results.json      # Saved baseline scores
├── models.py                  # Pydantic Action + Observation models
├── data_generator.py          # Generates synthetic financial statements
├── fraud_injector.py          # Injects fraud patterns
├── grader.py                  # Scores agent 0.0–1.0
├── client.py                  # OpenEnv client
├── openenv.yaml               # OpenEnv spec config
└── server/
    ├── app.py                 # FastAPI server
    ├── ff_env_environment.py  # reset() / step() / state()
    └── Dockerfile             # Container definition
```

---

## Design Notes

**Synthetic data** — all statements are procedurally generated with deterministic seeds. No real company data used.

**Fraud types** are based on real SEC enforcement cases — Enron (revenue inflation), WorldCom (expense hiding), Sunbeam (channel stuffing).

**Hard task is genuinely hard** — 3 layered frauds, red herrings, requires cross-statement reasoning and benchmark comparison. Frontier LLMs score ~0.55.