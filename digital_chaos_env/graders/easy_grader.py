"""Easy task grader."""


def grade_episode(episode: dict) -> float:
    metrics = episode.get("metrics", {})
    return float(metrics.get("final_score", 0.0))
