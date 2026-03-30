import json
import os
import random
import statistics

from digital_chaos_env import DigitalChaosAction, DigitalChaosEnv
from digital_chaos_env.models import ActionType

TASK_IDS = ["easy", "medium", "hard"]


def heuristic_action(observation):
    notifications = observation.get("notifications", [])

    # Important notifications first
    for note in notifications:
        if note.get("important"):
            return DigitalChaosAction(action_type=ActionType.check_notification, target_id=note["id"])

    # Ignore useless notifications
    for note in notifications:
        if not note.get("important"):
            return DigitalChaosAction(action_type=ActionType.ignore_notification, target_id=note["id"])

    # Complete highest-priority pending task
    pending = []
    for task in observation.get("tasks", []):
        if not task.get("completed"):
            pending.append(task)

    if pending:
        pending.sort(key=lambda t: int(t["priority"]), reverse=True)
        return DigitalChaosAction(action_type=ActionType.complete_task, target_id=pending[0]["id"])

    return DigitalChaosAction(action_type=ActionType.delay_task)


def llm_action(observation, model):
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = {
        "instruction": "Pick exactly one valid next action for productivity.",
        "allowed_action_type": ["complete_task", "ignore_notification", "check_notification", "delay_task"],
        "observation": observation,
        "output_json_schema": {"action_type": "string", "target_id": "string|null"},
    }

    response = client.responses.create(
        model=model,
        temperature=0,
        input=[{"role": "user", "content": json.dumps(prompt)}],
    )

    text = response.output_text.strip()
    try:
        payload = json.loads(text)
        return DigitalChaosAction(
            action_type=payload.get("action_type", "delay_task"),
            target_id=payload.get("target_id"),
        )
    except Exception:
        return heuristic_action(observation)


def choose_action(observation, mode, model):
    if mode == "openai" and os.getenv("OPENAI_API_KEY"):
        return llm_action(observation, model)
    return heuristic_action(observation)


def random_action(observation, rng):
    actions = [DigitalChaosAction(action_type=ActionType.delay_task)]

    for task in observation.get("tasks", []):
        if not task.get("completed"):
            actions.append(DigitalChaosAction(action_type=ActionType.complete_task, target_id=task["id"]))

    for note in observation.get("notifications", []):
        actions.append(DigitalChaosAction(action_type=ActionType.ignore_notification, target_id=note["id"]))
        actions.append(DigitalChaosAction(action_type=ActionType.check_notification, target_id=note["id"]))

    return rng.choice(actions)


def run_single_task_http(env, task_id, mode, model):
    result = env.reset(task_id=task_id)
    total_reward = 0.0

    while not result.done:
        obs = result.observation.model_dump()
        action = choose_action(obs, mode, model)
        result = env.step(action)
        total_reward += result.reward

    grade = env.grader()
    return {
        "task_id": task_id,
        "score": grade["final_score"],
        "metrics": grade,
        "total_reward": round(total_reward, 4),
    }


def run_baseline(base_url="http://127.0.0.1:8000"):
    env = DigitalChaosEnv(base_url=base_url)

    preferred_mode = os.getenv("DIGITAL_CHAOS_BASELINE_MODE", "heuristic").strip().lower()
    model = os.getenv("DIGITAL_CHAOS_BASELINE_MODEL", "gpt-4.1-mini")
    mode = "openai" if preferred_mode == "openai" and os.getenv("OPENAI_API_KEY") else "heuristic"

    results = []
    for task_id in TASK_IDS:
        results.append(run_single_task_http(env, task_id, mode, model))

    avg_score = sum(item["score"] for item in results) / len(results)

    return {
        "baseline": f"{mode}_policy",
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "model": model if mode == "openai" else None,
        "results": results,
        "average_score": round(avg_score, 4),
    }


def run_baseline_local():
    from digital_chaos_env.server.digital_chaos_env_environment import DigitalChaosEnvironment

    env = DigitalChaosEnvironment()
    results = []

    for task_id in TASK_IDS:
        observation, _, done, _ = env.reset(task_id=task_id)
        total_reward = 0.0

        while not done:
            action = heuristic_action(observation.model_dump())
            observation, reward, done, _ = env.step(action)
            total_reward += reward

        grade = env.grader()
        results.append(
            {
                "task_id": task_id,
                "score": grade["final_score"],
                "metrics": grade,
                "total_reward": round(total_reward, 4),
            }
        )

    avg_score = sum(item["score"] for item in results) / len(results)

    return {
        "baseline": "heuristic_policy",
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "results": results,
        "average_score": round(avg_score, 4),
    }


def summarize(values):
    return {
        "mean": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "stdev": round(statistics.pstdev(values), 4),
    }


def run_benchmark_local(seed=42, runs=3):
    from digital_chaos_env.server.digital_chaos_env_environment import DigitalChaosEnvironment

    runs = max(1, min(10, int(runs)))
    rng = random.Random(seed)

    heuristic_scores = []
    random_scores = []

    for run_index in range(runs):
        env = DigitalChaosEnvironment()
        one_run_scores = []

        for task_id in TASK_IDS:
            observation, _, done, _ = env.reset(task_id=task_id, seed=seed + run_index)
            while not done:
                action = heuristic_action(observation.model_dump())
                observation, _, done, _ = env.step(action)
            one_run_scores.append(float(env.grader()["final_score"]))

        heuristic_scores.append(sum(one_run_scores) / len(one_run_scores))

        env = DigitalChaosEnvironment()
        one_run_scores = []

        for task_id in TASK_IDS:
            observation, _, done, _ = env.reset(task_id=task_id, seed=seed + run_index)
            while not done:
                action = random_action(observation.model_dump(), rng)
                observation, _, done, _ = env.step(action)
            one_run_scores.append(float(env.grader()["final_score"]))

        random_scores.append(sum(one_run_scores) / len(one_run_scores))

    heuristic_mean = sum(heuristic_scores) / len(heuristic_scores)
    random_mean = sum(random_scores) / len(random_scores)

    return {
        "seed": seed,
        "runs": runs,
        "policies": {
            "heuristic": summarize(heuristic_scores),
            "random": summarize(random_scores),
        },
        "winner": "heuristic" if heuristic_mean >= random_mean else "random",
    }


def main():
    base_url = os.getenv("DIGITAL_CHAOS_BASE_URL", "http://127.0.0.1:8000")
    result = run_baseline(base_url=base_url)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
