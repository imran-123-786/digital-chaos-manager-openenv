from __future__ import annotations

from typing import Any


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def clamp_open01(value: float, eps: float = 0.01) -> float:
    """Clamp score strictly inside (0, 1) for validator compliance."""
    return max(eps, min(1.0 - eps, value))


def compute_grader_score(metrics: dict[str, Any]) -> dict[str, float]:
    total_tasks = max(1, int(metrics.get("total_tasks", 0)))
    completed_tasks = int(metrics.get("completed_tasks", 0))
    completion_score = clamp_open01(clamp01(completed_tasks / total_tasks))

    order_total = max(1, int(metrics.get("priority_events", 0)))
    order_violations = int(metrics.get("priority_violations", 0))
    priority_score = clamp_open01(clamp01(1.0 - (order_violations / order_total)))

    useless_seen = max(1, int(metrics.get("useless_notifications_seen", 0)))
    ignored_useless = int(metrics.get("ignored_useless", 0))
    checked_useless = int(metrics.get("checked_useless", 0))
    missed_important = int(metrics.get("missed_important", 0))
    distraction_raw = (ignored_useless - 0.5 * checked_useless - 0.7 * missed_important) / useless_seen
    distraction_score = clamp_open01(clamp01(distraction_raw))

    steps_used = max(1, int(metrics.get("steps_used", 0)))
    time_budget = max(1, int(metrics.get("time_budget", 0)))
    pace_score = clamp_open01(clamp01((completed_tasks / steps_used) * (time_budget / max(steps_used, 1))))
    efficiency_score = clamp_open01(clamp01(0.5 * completion_score + 0.5 * pace_score))

    final_score = clamp_open01(clamp01(
        0.4 * completion_score
        + 0.3 * priority_score
        + 0.2 * distraction_score
        + 0.1 * efficiency_score
    ))

    return {
        "task_completion": round(completion_score, 4),
        "priority_handling": round(priority_score, 4),
        "distraction_control": round(distraction_score, 4),
        "efficiency": round(efficiency_score, 4),
        "final_score": round(final_score, 4),
    }
