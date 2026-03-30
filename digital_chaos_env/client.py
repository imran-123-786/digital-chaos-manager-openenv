from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import DigitalChaosAction, DigitalChaosObservation


@dataclass
class DigitalChaosStepResult:
    observation: DigitalChaosObservation
    reward: float
    done: bool
    info: dict[str, Any]


class DigitalChaosEnv:
    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str = "easy") -> DigitalChaosStepResult:
        import requests

        response = requests.get(f"{self.base_url}/reset", params={"task_id": task_id}, timeout=30)
        response.raise_for_status()
        return self._parse_payload(response.json())

    def step(self, action: DigitalChaosAction) -> DigitalChaosStepResult:
        import requests

        response = requests.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            timeout=30,
        )
        response.raise_for_status()
        return self._parse_payload(response.json())

    def state(self) -> dict[str, Any]:
        import requests

        response = requests.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return response.json()

    def tasks(self) -> dict[str, Any]:
        import requests

        response = requests.get(f"{self.base_url}/tasks", timeout=30)
        response.raise_for_status()
        return response.json()

    def grader(self) -> dict[str, float]:
        import requests

        response = requests.get(f"{self.base_url}/grader", timeout=30)
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        return None

    def _parse_payload(self, payload: dict[str, Any]) -> DigitalChaosStepResult:
        observation = DigitalChaosObservation(**payload["observation"])
        return DigitalChaosStepResult(
            observation=observation,
            reward=float(payload["reward"]),
            done=bool(payload["done"]),
            info=dict(payload.get("info", {})),
        )
