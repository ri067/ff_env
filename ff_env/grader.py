"""
grader.py

Scores the agent's performance for a completed episode.
Returns a float between 0.0 and 1.0.

Grading is fully deterministic — same flags, same score, every time.
This is what the hackathon judges will run to evaluate submissions.

Scoring breakdown:
  - Fraud detection (core):     60% — did agent find all injected frauds?
  - Precision (no false alarms): 20% — did agent avoid flagging clean data?
  - Efficiency:                  20% — did agent solve it without wasting steps?
"""


def grade_episode(
    injected_frauds: list[dict],
    flags_raised:    list[dict],
    steps_used:      int,
    step_budget:     int,
) -> tuple[float, dict]:
    """
    Grade a completed episode.

    Args:
        injected_frauds: ground truth — list of fraud dicts from fraud_injector
        flags_raised:    what the agent flagged — list of flag dicts
        steps_used:      how many steps the agent used
        step_budget:     maximum steps allowed for this task

    Returns:
        (score, breakdown) where score is 0.0–1.0 and breakdown is a dict
        explaining how the score was computed.
    """

    num_injected = len(injected_frauds)
    if num_injected == 0:
        # Edge case: no frauds injected (shouldn't happen in normal tasks)
        return 1.0, {"note": "No frauds injected — trivially clean"}

    injected_types = {f["type"] for f in injected_frauds}

    # ── 1. Fraud Detection Score (60%) ────────────────────────
    # How many of the injected frauds did the agent correctly identify?

    correct_flags  = [f for f in flags_raised if f.get("correct", False)]
    detected_types = {f["fraud_type"] for f in correct_flags}

    frauds_found   = len(detected_types & injected_types)
    detection_score = frauds_found / num_injected  # 0.0 to 1.0

    # Bonus: correct line item identified
    line_item_bonus = 0.0
    for flag in correct_flags:
        injected = next(
            (f for f in injected_frauds if f["type"] == flag["fraud_type"]),
            None
        )
        if injected and flag.get("line_item", "") == injected.get("line_item", ""):
            line_item_bonus += (1.0 / num_injected) * 0.10  # up to 10% bonus

    # ── 2. Precision Score (20%) ──────────────────────────────
    # Penalise false positives — flagging things that aren't fraud

    false_positives  = [f for f in flags_raised if not f.get("correct", False)]
    num_fp           = len(false_positives)

    if num_fp == 0:
        precision_score = 1.0
    elif num_fp == 1:
        precision_score = 0.5
    else:
        precision_score = max(0.0, 1.0 - (num_fp * 0.4))

    # ── 3. Efficiency Score (20%) ─────────────────────────────
    # Reward solving quickly — penalise using nearly all steps

    if steps_used <= 0:
        efficiency_score = 1.0
    else:
        steps_fraction   = steps_used / step_budget
        efficiency_score = max(0.0, 1.0 - steps_fraction * 0.6)
        # Agent uses 50% of budget → 0.7 efficiency
        # Agent uses 100% of budget → 0.4 efficiency
        # Agent uses 30% of budget → 0.82 efficiency

    # ── Final Score ───────────────────────────────────────────

    raw_score = (
        detection_score  * 0.60 +
        line_item_bonus         +   # up to 0.10 extra
        precision_score  * 0.20 +
        efficiency_score * 0.20
    )

    final_score = round(min(1.0, max(0.0, raw_score)), 4)

    breakdown = {
        "frauds_injected":   num_injected,
        "frauds_found":      frauds_found,
        "detection_score":   round(detection_score, 4),
        "line_item_bonus":   round(line_item_bonus, 4),
        "false_positives":   num_fp,
        "precision_score":   round(precision_score, 4),
        "steps_used":        steps_used,
        "step_budget":       step_budget,
        "efficiency_score":  round(efficiency_score, 4),
        "final_score":       final_score,
    }

    return final_score, breakdown


# ─────────────────────────────────────────────────────────────
# Quick tests — run directly to verify grader logic
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    INJECTED = [
        {"type": "revenue_inflation", "line_item": "revenue",   "statement": "income_statement"},
        {"type": "expense_hiding",    "line_item": "operating_expenses", "statement": "income_statement"},
    ]

    print("=" * 50)
    print("TEST 1: Perfect agent — finds both, no false positives, fast")
    score, breakdown = grade_episode(
        injected_frauds=INJECTED,
        flags_raised=[
            {"fraud_type": "revenue_inflation", "line_item": "revenue",   "correct": True},
            {"fraud_type": "expense_hiding",    "line_item": "operating_expenses", "correct": True},
        ],
        steps_used=6,
        step_budget=20,
    )
    print(f"Score: {score}")
    print(f"Breakdown: {breakdown}\n")

    print("=" * 50)
    print("TEST 2: Partial — finds one fraud, misses other, no false positives")
    score, breakdown = grade_episode(
        injected_frauds=INJECTED,
        flags_raised=[
            {"fraud_type": "revenue_inflation", "line_item": "revenue", "correct": True},
        ],
        steps_used=15,
        step_budget=20,
    )
    print(f"Score: {score}")
    print(f"Breakdown: {breakdown}\n")

    print("=" * 50)
    print("TEST 3: Wrong agent — finds nothing, raises 2 false positives")
    score, breakdown = grade_episode(
        injected_frauds=INJECTED,
        flags_raised=[
            {"fraud_type": "channel_stuffing",   "line_item": "receivables", "correct": False},
            {"fraud_type": "asset_overstatement","line_item": "total_assets", "correct": False},
        ],
        steps_used=20,
        step_budget=20,
    )
    print(f"Score: {score}")
    print(f"Breakdown: {breakdown}\n")

    print("=" * 50)
    print("TEST 4: Finds all, but one false positive, used all steps")
    score, breakdown = grade_episode(
        injected_frauds=INJECTED,
        flags_raised=[
            {"fraud_type": "revenue_inflation", "line_item": "revenue",   "correct": True},
            {"fraud_type": "expense_hiding",    "line_item": "operating_expenses", "correct": True},
            {"fraud_type": "channel_stuffing",  "line_item": "receivables", "correct": False},
        ],
        steps_used=20,
        step_budget=20,
    )
    print(f"Score: {score}")
    print(f"Breakdown: {breakdown}")