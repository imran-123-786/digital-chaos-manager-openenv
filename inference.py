import json
import os
from typing import Any

from openai import OpenAI

from digital_chaos_env.client import DigitalChaosEnv
from digital_chaos_env.models import ActionType, DigitalChaosAction

# Required env vars (as per submission checklist)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional when running custom/local image workflows
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

TASK_IDS = ["easy", "medium", "hard"]


def _log(tag: str, payload: dict[str, Any]) -> None:
    # Required structured stdout format
    print(f"{tag} {json.dumps(payload, ensure_ascii=False)}")


def _fallback_action(observation: dict[str, Any]) -> dict[str, Any]:
    notes = observation.get("notifications", [])
    for n in notes:
        if n.get("important"):
            return {"action_type": "check_notification", "target_id": n["id"]}
    for n in notes:
        if not n.get("important"):
            return {"action_type": "ignore_notification", "target_id": n["id"]}

    tasks = [t for t in observation.get("tasks", []) if not t.get("completed")]
    if tasks:
        tasks.sort(key=lambda x: int(x["priority"]), reverse=True)
        return {"action_type": "complete_task", "target_id": tasks[0]["id"]}
    return {"action_type": "delay_task", "target_id": None}


def _llm_action(client: OpenAI, observation: dict[str, Any]) -> dict[str, Any]:
    prompt = {
        "instruction": "Choose one valid next action. Return JSON only.",
        "allowed_action_type": [
            "complete_task",
            "ignore_notification",
            "check_notification",
            "delay_task",
        ],
        "observation": observation,
        "format": {"action_type": "string", "target_id": "string|null"},
    }
    response = client.responses.create(
        model=MODEL_NAME,
        temperature=0,
        input=[{"role": "user", "content": json.dumps(prompt)}],
    )
    text = response.output_text.strip()
    try:
        payload = json.loads(text)
        return {
            "action_type": payload.get("action_type", "delay_task"),
            "target_id": payload.get("target_id"),
        }
    except Exception:
        return _fallback_action(observation)


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required")

    env_base_url = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = DigitalChaosEnv(base_url=env_base_url)

    _log(
        "START",
        {
            "api_base_url": API_BASE_URL,
            "model_name": MODEL_NAME,
            "local_image_name": LOCAL_IMAGE_NAME,
            "env_base_url": env_base_url,
            "tasks": TASK_IDS,
        },
    )

    results = []
    for task_id in TASK_IDS:
        step_result = env.reset(task_id=task_id)
        total_reward = 0.0
        step_count = 0

        while not step_result.done:
            obs = step_result.observation.model_dump()
            action_dict = _llm_action(client, obs)
            action = DigitalChaosAction(
                action_type=ActionType(action_dict["action_type"]),
                target_id=action_dict.get("target_id"),
            )
            step_result = env.step(action)
            step_count += 1
            total_reward += step_result.reward
            _log(
                "STEP",
                {
                    "task_id": task_id,
                    "step": step_count,
                    "action_type": action.action_type.value,
                    "target_id": action.target_id,
                    "reward": step_result.reward,
                    "done": step_result.done,
                },
            )

        grade = env.grader()
        result = {
            "task_id": task_id,
            "score": grade["final_score"],
            "metrics": grade,
            "total_reward": round(total_reward, 4),
        }
        results.append(result)

    avg_score = sum(r["score"] for r in results) / len(results)
    _log("END", {"results": results, "average_score": round(avg_score, 4)})


if __name__ == "__main__":
    main()
