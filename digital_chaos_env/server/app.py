from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Query

from digital_chaos_env.models import DigitalChaosAction
from digital_chaos_env.server.digital_chaos_env_environment import DigitalChaosEnvironment
from digital_chaos_env.tasks import list_task_specs

app = FastAPI(title="Digital Chaos Manager", version="1.0.0")
env = DigitalChaosEnvironment()


def _payload(observation: Any, reward: float, done: bool, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/")
def root() -> dict[str, Any]:
    state_snapshot = env.state()
    return {
        "name": "Digital Chaos Manager",
        "status": "ok",
        "current_task_id": state_snapshot["task_id"],
        "difficulty": state_snapshot["difficulty"],
        "time_left": state_snapshot["time_left"],
        "quickstart": {
            "reset_easy": "/reset?task_id=easy",
            "docs": "/docs",
            "tasks": "/tasks",
            "baseline": "/baseline",
        },
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/reset")
def reset(
    task_id: str = Query(default="easy"),
    profile: str = Query(default="office"),
    seed: int | None = Query(default=None),
) -> dict[str, Any]:
    observation, reward, done, info = env.reset(task_id, profile=profile, seed=seed)
    return _payload(observation, reward, done, info)


@app.post("/reset")
def reset_post(
    task_id: str = Query(default="easy"),
    profile: str = Query(default="office"),
    seed: int | None = Query(default=None),
) -> dict[str, Any]:
    observation, reward, done, info = env.reset(task_id, profile=profile, seed=seed)
    return _payload(observation, reward, done, info)


@app.post("/step")
def step(action: DigitalChaosAction) -> dict[str, Any]:
    observation, reward, done, info = env.step(action)
    return _payload(observation, reward, done, info)


@app.get("/state")
def state() -> dict[str, Any]:
    return env.state()


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {
        "tasks": list_task_specs(),
        "action_schema": {
            "action_type": ["complete_task", "ignore_notification", "check_notification", "delay_task"],
            "target_id": "optional string; required for all except delay_task",
        },
    }


@app.get("/grader")
def grader() -> dict[str, Any]:
    return env.grader()


@app.get("/baseline")
def baseline() -> dict[str, Any]:
    from digital_chaos_env.baseline import run_baseline_local

    return run_baseline_local()


@app.get("/benchmark")
def benchmark(seed: int = Query(default=42), runs: int = Query(default=3)) -> dict[str, Any]:
    from digital_chaos_env.baseline import run_benchmark_local

    return run_benchmark_local(seed=seed, runs=runs)


@app.get("/explain")
def explain() -> dict[str, Any]:
    return env.explain()


@app.get("/analytics")
def analytics() -> dict[str, Any]:
    return env.analytics()


def main() -> None:
    import uvicorn

    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
