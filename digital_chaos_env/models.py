from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    complete_task = "complete_task"
    ignore_notification = "ignore_notification"
    check_notification = "check_notification"
    delay_task = "delay_task"


class TaskPriority(int, Enum):
    low = 1
    medium = 2
    high = 3


class TaskItem(BaseModel):
    id: str
    title: str
    priority: TaskPriority
    completed: bool = False


class NotificationItem(BaseModel):
    id: str
    message: str
    important: bool
    active: bool = True
    checked: bool = False
    ignored: bool = False


class DigitalChaosAction(BaseModel):
    action_type: ActionType
    target_id: str | None = None


class DigitalChaosObservation(BaseModel):
    task_id: str
    difficulty: str
    time_left: int
    step_count: int
    tasks: list[TaskItem]
    notifications: list[NotificationItem]


class DigitalChaosReward(BaseModel):
    value: float
    completion: float = 0.0
    distraction: float = 0.0
    penalty: float = 0.0


class DigitalChaosStepPayload(BaseModel):
    observation: DigitalChaosObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
