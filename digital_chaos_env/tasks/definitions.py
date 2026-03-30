from __future__ import annotations

from dataclasses import dataclass

from digital_chaos_env.models import TaskPriority


@dataclass(frozen=True)
class TaskTemplate:
    id: str
    title: str
    priority: TaskPriority


@dataclass(frozen=True)
class NotificationTemplate:
    id: str
    message: str
    important: bool
    arrives_at_step: int


TASK_DEFINITIONS: dict[str, dict] = {
    "easy": {
        "name": "Focused Morning Sprint",
        "difficulty": "easy",
        "time_budget": 6,
        "tasks": [
            TaskTemplate("t1", "Reply to manager update", TaskPriority.high),
            TaskTemplate("t2", "Prepare standup notes", TaskPriority.medium),
            TaskTemplate("t3", "Clean inbox labels", TaskPriority.low),
        ],
        "notifications": [],
    },
    "medium": {
        "name": "Mixed Priority Office Block",
        "difficulty": "medium",
        "time_budget": 8,
        "tasks": [
            TaskTemplate("t1", "Resolve customer escalation", TaskPriority.high),
            TaskTemplate("t2", "Review PR for teammate", TaskPriority.medium),
            TaskTemplate("t3", "Update status dashboard", TaskPriority.medium),
            TaskTemplate("t4", "Organize bookmarks", TaskPriority.low),
        ],
        "notifications": [
            NotificationTemplate("n1", "Production alert: elevated errors", True, 1),
            NotificationTemplate("n2", "Flash sale ad popup", False, 2),
            NotificationTemplate("n3", "Reminder: submit daily report", True, 4),
            NotificationTemplate("n4", "Social app mention", False, 5),
        ],
    },
    "hard": {
        "name": "Chaos Afternoon Crunch",
        "difficulty": "hard",
        "time_budget": 10,
        "tasks": [
            TaskTemplate("t1", "Handle executive incident brief", TaskPriority.high),
            TaskTemplate("t2", "Fix broken release checklist", TaskPriority.high),
            TaskTemplate("t3", "Review two urgent bug tickets", TaskPriority.medium),
            TaskTemplate("t4", "Plan tomorrow sprint goals", TaskPriority.medium),
            TaskTemplate("t5", "Sort desktop files", TaskPriority.low),
            TaskTemplate("t6", "Refactor old notes", TaskPriority.low),
        ],
        "notifications": [
            NotificationTemplate("n1", "PagerDuty: API latency spike", True, 1),
            NotificationTemplate("n2", "Meme in team chat", False, 1),
            NotificationTemplate("n3", "Security patch reminder", True, 2),
            NotificationTemplate("n4", "Promo newsletter", False, 3),
            NotificationTemplate("n5", "Client follow-up ping", True, 5),
            NotificationTemplate("n6", "Streaming app recommendation", False, 6),
            NotificationTemplate("n7", "Calendar conflict alert", True, 7),
        ],
    },
}


def list_task_specs() -> list[dict]:
    specs: list[dict] = []
    for task_id, cfg in TASK_DEFINITIONS.items():
        specs.append(
            {
                "task_id": task_id,
                "name": cfg["name"],
                "difficulty": cfg["difficulty"],
                "time_budget": cfg["time_budget"],
                "task_count": len(cfg["tasks"]),
                "notification_count": len(cfg["notifications"]),
            }
        )
    return specs
