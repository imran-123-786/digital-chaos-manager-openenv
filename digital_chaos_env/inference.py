import json
import os

from openai import OpenAI

from digital_chaos_env.client import DigitalChaosEnv
from digital_chaos_env.models import ActionType, DigitalChaosAction

TASK_IDS = ["easy", "medium", "hard"]
TASK_GRADERS = {
    "easy": "graders/easy_grader.py",
    "medium": "graders/medium_grader.py",
    "hard": "graders/hard_grader.py",
}


def get_required_env(name):
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def fallback_action(observation):
    notes = observation.get("notifications", [])
    for n in notes:
        if n.get("important"):
            return {"action_type": "check_notification", "target_id": n["id"]}
    for n in notes:
        if not n.get("important"):
            return {"action_type": "ignore_notification", "target_id": n["id"]}

    pending = [t for t in observation.get("tasks", []) if not t.get("completed")]
    if pending:
        pending.sort(key=lambda x: int(x["priority"]), reverse=True)
        return {"action_type": "complete_task", "target_id": pending[0]["id"]}

    return {"action_type": "delay_task", "target_id": None}


def llm_action(client, model_name, observation):
    prompt = {
        "instruction": "Choose one valid next action. Return JSON only.",
        "allowed_action_type": ["complete_task", "ignore_notification", "check_notification", "delay_task"],
        "observation": observation,
        "format": {"action_type": "string", "target_id": "string|null"},
    }

    resp = client.responses.create(
        model=model_name,
        temperature=0,
        input=[{"role": "user", "content": json.dumps(prompt)}],
    )

    text = resp.output_text.strip()
    try:
        payload = json.loads(text)
        action_type = payload.get("action_type", "delay_task")
        target_id = payload.get("target_id")
        return {"action_type": action_type, "target_id": target_id}
    except Exception:
        return fallback_action(observation)


def run_episode(env, client, model_name, task_id):
    step_result = env.reset(task_id=task_id)
    total_reward = 0.0

    while not step_result.done:
        obs = step_result.observation.model_dump()
        action_dict = llm_action(client, model_name, obs)
        action = DigitalChaosAction(
            action_type=ActionType(action_dict["action_type"]),
            target_id=action_dict.get("target_id"),
        )
        step_result = env.step(action)
        total_reward += step_result.reward

    grade = env.grader()
    return {
        "task_id": task_id,
        "score": grade["final_score"],
        "metrics": grade,
        "total_reward": round(total_reward, 4),
    }


def main():
    api_base_url = get_required_env("API_BASE_URL")
    model_name = get_required_env("MODEL_NAME")
    hf_token = get_required_env("HF_TOKEN")

    env_base_url = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    env = DigitalChaosEnv(base_url=env_base_url)

    print(
        f"[START] summary=run api_base_url={api_base_url} model_name={model_name} env_base_url={env_base_url}",
        flush=True,
    )

    results = []
    for task_id in TASK_IDS:
        print(f"[START] task={task_id} grader={TASK_GRADERS[task_id]}", flush=True)
        results.append(run_episode(env, client, model_name, task_id))
        latest = results[-1]
        print(
            f"[END] task={task_id} grader={TASK_GRADERS[task_id]} score={latest['score']} total_reward={latest['total_reward']}",
            flush=True,
        )

    avg_score = sum(item["score"] for item in results) / len(results)

    output = {
        "inference": "openai-client",
        "api_base_url": api_base_url,
        "model_name": model_name,
        "results": results,
        "average_score": round(avg_score, 4),
    }

    print(f"[END] summary=overall average_score={round(avg_score, 4)} tasks={len(results)}", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
