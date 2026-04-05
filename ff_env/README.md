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

An OpenEnv-compatible RL environment where an AI agent plays the role of a **forensic accountant** auditing synthetic company financial statements to detect accounting fraud.

## Overview

The agent receives 4 quarters of financial data across 3 statements for a fictitious company. One or more fraud patterns have been secretly injected. The agent must investigate, identify the fraud, and submit a report within a step budget.

## Fraud Patterns

| Fraud Type | Description | Key Signal |
|---|---|---|
| `revenue_inflation` | Fake sales recorded | Revenue spikes but cash flow stays flat |
| `expense_hiding` | Costs moved off books | Expenses drop with no explanation |
| `channel_stuffing` | Excess goods shipped | Receivables grow faster than revenue |
| `earnings_smoothing` | Profits manipulated | Unnaturally consistent quarterly growth |
| `asset_overstatement` | Assets inflated | Assets grow much faster than revenue |

## Tasks

| Task | Difficulty | Frauds | Steps | LLM Score |
|---|---|---|---|---|
| `easy` | Easy | 1 | 15 | ~1.0 |
| `medium` | Medium | 2 | 20 | ~0.85 |
| `hard` | Hard | 3 | 25 | ~0.55 |

## Action Space

| Action | Description |
|---|---|
| `inspect` | View a metric across all 4 quarters |
| `compare` | Cross-reference two metrics |
| `flag` | Raise a fraud alert |
| `check_benchmark` | Compare against industry average |
| `request_detail` | Full breakdown of one quarter |
| `submit_report` | Finalise findings, end episode |

## Quick Start
```bash
# Test the live endpoint
curl -X POST https://OrangeUnknown-ff-env.hf.space/reset
```

## Run Baseline Inference
```bash
git clone https://huggingface.co/spaces/OrangeUnknown/ff-env
cd ff-env
pip install openenv-core pydantic openai

export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="your-groq-key"

python3 inference.py
```

## Baseline Results (llama-3.3-70b-versatile)

| Task | Score | Steps |
|---|---|---|
| Easy | 1.000 | 6 |
| Medium | 1.000 | 9 |
| Hard | 0.573 | 14 |
| **Average** | **0.858** | |

## Reward Function

- Correct fraud flagged: +0.25
- Correct line item: +0.10
- Relevant comparison: +0.10
- False positive: -0.10
- Final score (0.0-1.0): detection 60% + precision 20% + efficiency 20%