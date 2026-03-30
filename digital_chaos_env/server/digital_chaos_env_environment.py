import random

from digital_chaos_env.graders import compute_grader_score
from digital_chaos_env.models import (
    ActionType,
    DigitalChaosAction,
    DigitalChaosObservation,
    DigitalChaosReward,
    NotificationItem,
    TaskItem,
)
from digital_chaos_env.tasks import TASK_DEFINITIONS


class DigitalChaosEnvironment:
    def __init__(self):
        self._task_id = "easy"
        self._state = {}
        self.reset("easy")

    def reset(self, task_id="easy", profile="office", seed=None):
        if task_id not in TASK_DEFINITIONS:
            task_id = "easy"

        if profile not in {"office", "student", "support"}:
            profile = "office"

        cfg = TASK_DEFINITIONS[task_id]
        self._task_id = task_id

        rng = random.Random(seed)
        task_templates = list(cfg["tasks"])
        note_templates = list(cfg["notifications"])
        rng.shuffle(task_templates)
        rng.shuffle(note_templates)

        tasks = [TaskItem(id=t.id, title=t.title, priority=t.priority, completed=False) for t in task_templates]
        notes = [
            NotificationItem(
                id=n.id,
                message=n.message,
                important=n.important,
                active=False,
                checked=False,
                ignored=False,
            )
            for n in note_templates
        ]

        time_budget = int(cfg["time_budget"])
        if profile == "student":
            time_budget = max(4, time_budget - 1)
        elif profile == "support":
            time_budget = time_budget + 1

        self._state = {
            "task_id": task_id,
            "profile": profile,
            "seed": seed,
            "difficulty": cfg["difficulty"],
            "time_budget": time_budget,
            "time_left": time_budget,
            "step_count": 0,
            "tasks": tasks,
            "notifications": notes,
            "notification_schedule": {n.id: n.arrives_at_step for n in cfg["notifications"]},
            "last_action_type": None,
            "context_switches": 0,
            "explanations": [],
            "metrics": {
                "total_tasks": len(tasks),
                "completed_tasks": 0,
                "priority_events": 0,
                "priority_violations": 0,
                "useless_notifications_seen": 0,
                "ignored_useless": 0,
                "checked_useless": 0,
                "important_notifications_seen": 0,
                "checked_important": 0,
                "missed_important": 0,
                "steps_used": 0,
                "time_budget": time_budget,
            },
        }

        self._activate_notifications(0)

        info = {
            "message": "environment_reset",
            "task_id": task_id,
            "profile": profile,
            "seed": seed,
        }
        return self._observation(), 0.0, False, info

    def step(self, action: DigitalChaosAction):
        if self._state["time_left"] <= 0:
            return self._observation(), 0.0, True, {"error": "episode_done", "grader": self.grader()}

        self._state["step_count"] += 1
        self._state["metrics"]["steps_used"] = self._state["step_count"]
        self._activate_notifications(self._state["step_count"])

        prev_action = self._state["last_action_type"]
        current_action = action.action_type.value
        if prev_action and prev_action != current_action:
            self._state["context_switches"] += 1
        self._state["last_action_type"] = current_action

        reward = DigitalChaosReward(value=0.0)
        info = {"action": current_action, "target_id": action.target_id}

        if action.action_type == ActionType.complete_task:
            self._complete_task(action, reward, info)
        elif action.action_type == ActionType.ignore_notification:
            self._ignore_notification(action, reward, info)
        elif action.action_type == ActionType.check_notification:
            self._check_notification(action, reward, info)
        else:
            reward.penalty -= 0.2
            reward.value -= 0.2
            info["result"] = "task_delayed"

        self._state["time_left"] -= 1
        done = self._is_done()
        if done:
            self._mark_missed_important_notifications()
            info["grader"] = self.grader()

        explanation = {
            "step": self._state["step_count"],
            "action": current_action,
            "result": info.get("result", "none"),
            "reward": reward.value,
            "time_left": self._state["time_left"],
        }
        self._state["explanations"].append(explanation)

        info["reward_breakdown"] = reward.model_dump()
        info["explanation"] = explanation
        return self._observation(), reward.value, done, info

    def state(self):
        return {
            "task_id": self._state["task_id"],
            "profile": self._state["profile"],
            "seed": self._state["seed"],
            "difficulty": self._state["difficulty"],
            "time_left": self._state["time_left"],
            "step_count": self._state["step_count"],
            "observation": self._observation().model_dump(),
            "metrics": dict(self._state["metrics"]),
            "grader": self.grader(),
        }

    def grader(self):
        return compute_grader_score(self._state["metrics"])

    def explain(self):
        history = self._state["explanations"]
        return {
            "last_explanation": history[-1] if history else None,
            "history_count": len(history),
            "history_tail": history[-5:],
        }

    def analytics(self):
        steps_used = int(self._state["metrics"].get("steps_used", 0))
        completed = int(self._state["metrics"].get("completed_tasks", 0))
        safe_steps = max(1, steps_used)

        return {
            "context_switches": self._state["context_switches"],
            "focus_ratio": round(completed / safe_steps, 4),
            "completed_tasks": completed,
            "steps_used": steps_used,
            "profile": self._state["profile"],
            "seed": self._state["seed"],
        }

    def _observation(self):
        active_notes = [n for n in self._state["notifications"] if n.active]
        return DigitalChaosObservation(
            task_id=self._state["task_id"],
            difficulty=self._state["difficulty"],
            time_left=self._state["time_left"],
            step_count=self._state["step_count"],
            tasks=[t.model_copy(deep=True) for t in self._state["tasks"]],
            notifications=[n.model_copy(deep=True) for n in active_notes],
        )

    def _is_done(self):
        all_done = all(t.completed for t in self._state["tasks"])
        return self._state["time_left"] <= 0 or all_done

    def _activate_notifications(self, step):
        for note in self._state["notifications"]:
            if note.active:
                continue
            if self._state["notification_schedule"].get(note.id) == step:
                note.active = True
                if note.important:
                    self._state["metrics"]["important_notifications_seen"] += 1
                else:
                    self._state["metrics"]["useless_notifications_seen"] += 1

    def _complete_task(self, action, reward, info):
        task = self._find_task(action.target_id)
        if task is None:
            reward.penalty -= 0.2
            reward.value -= 0.2
            info["result"] = "invalid_task_id"
            return

        if task.completed:
            reward.penalty -= 0.2
            reward.value -= 0.2
            info["result"] = "task_already_completed"
            return

        if task.priority == 3:
            reward.completion += 1.0
            reward.value += 1.0
        elif task.priority == 2:
            reward.completion += 0.5
            reward.value += 0.5
        else:
            reward.completion += 0.2
            reward.value += 0.2

        task.completed = True
        self._state["metrics"]["completed_tasks"] += 1
        self._state["metrics"]["priority_events"] += 1

        if self._higher_priority_pending(task.priority):
            self._state["metrics"]["priority_violations"] += 1

        info["result"] = "task_completed"

    def _ignore_notification(self, action, reward, info):
        note = self._find_active_notification(action.target_id)
        if note is None:
            reward.penalty -= 0.1
            reward.value -= 0.1
            info["result"] = "invalid_notification_id"
            return

        if note.ignored or note.checked:
            reward.penalty -= 0.1
            reward.value -= 0.1
            info["result"] = "notification_already_handled"
            return

        note.ignored = True
        note.active = False

        if note.important:
            self._state["metrics"]["missed_important"] += 1
            reward.penalty -= 0.2
            reward.value -= 0.2
        else:
            self._state["metrics"]["ignored_useless"] += 1
            reward.distraction += 0.3
            reward.value += 0.3

        info["result"] = "notification_ignored"

    def _check_notification(self, action, reward, info):
        note = self._find_active_notification(action.target_id)
        if note is None:
            reward.penalty -= 0.1
            reward.value -= 0.1
            info["result"] = "invalid_notification_id"
            return

        if note.ignored or note.checked:
            reward.penalty -= 0.1
            reward.value -= 0.1
            info["result"] = "notification_already_handled"
            return

        note.checked = True
        note.active = False

        if note.important:
            self._state["metrics"]["checked_important"] += 1
            reward.distraction += 0.2
            reward.value += 0.2
        else:
            self._state["metrics"]["checked_useless"] += 1
            reward.penalty -= 0.1
            reward.value -= 0.1

        info["result"] = "notification_checked"

    def _mark_missed_important_notifications(self):
        for note in self._state["notifications"]:
            if note.important and note.active and not note.checked and not note.ignored:
                self._state["metrics"]["missed_important"] += 1
                note.active = False

    def _find_task(self, task_id):
        if not task_id:
            return None
        for task in self._state["tasks"]:
            if task.id == task_id:
                return task
        return None

    def _find_active_notification(self, note_id):
        if not note_id:
            return None
        for note in self._state["notifications"]:
            if note.id == note_id and note.active:
                return note
        return None

    def _higher_priority_pending(self, current_priority):
        for task in self._state["tasks"]:
            if not task.completed and int(task.priority) > int(current_priority):
                return True
        return False
