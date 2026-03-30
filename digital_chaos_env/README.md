---
title: Digital Chaos Manager
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Digital Chaos Manager (OpenEnv Environment)

Digital Chaos Manager is a real-world productivity simulation where an agent learns to manage tasks under constant digital distraction.

## Why this environment

This environment models office/student workflow pressure:
- Prioritize high-value tasks
- Handle useful vs useless notifications
- Avoid low-value actions under time pressure

## Action space

`DigitalChaosAction`:
- `action_type`: one of `complete_task`, `ignore_notification`, `check_notification`, `delay_task`
- `target_id`: task or notification id (optional only for `delay_task`)

## Observation space

`DigitalChaosObservation`:
- `task_id`, `difficulty`
- `time_left`, `step_count`
- `tasks[]` with priority and completion state
- `notifications[]` currently active

## Difficulty tasks

- `easy`: minimal distractions, basic scheduling behavior
- `medium`: mixed priorities + intermittent notifications
- `hard`: heavy interruptions + limited time budget

## Reward shaping

- Complete high task: `+1.0`
- Complete medium task: `+0.5`
- Complete low task: `+0.2`
- Ignore useless notification: `+0.3`
- Check useless notification: `-0.1`
- Ignore/miss important notification: `-0.2`
- Delay task: `-0.2`

## Grader

`/grader` returns deterministic `0.0?1.0` scores with weighted final score:
- Task completion: `0.4`
- Priority handling: `0.3`
- Distraction control: `0.2`
- Efficiency: `0.1`

## API endpoints

- `GET /reset?task_id=easy|medium|hard`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `GET /grader`
- `GET /baseline`
- `GET /explain`
- `GET /analytics`
- `GET /health`

## Advanced but simple features

- **Seeded reproducibility**: `GET /reset?task_id=medium&seed=123` reproduces the same scenario order.
- **Profiles**: `profile=office|student|support` adjusts episode pressure in a lightweight way.
- **Step explainability**: each `/step` returns `info.explanation` with action result and reward impact.
- **Decision trace endpoint**: `GET /explain` returns last explanation and recent explanation history.
- **Behavior analytics endpoint**: `GET /analytics` returns focus ratio and context-switch summary.

## Local run

```bash
pip install -e .
uvicorn server.app:app --reload
```

## Baseline script

```bash
python baseline.py
```

Modes:
- Default deterministic baseline: heuristic policy
- OpenAI baseline mode: set `OPENAI_API_KEY` and `DIGITAL_CHAOS_BASELINE_MODE=openai`

Example:

```bash
set OPENAI_API_KEY=your_key
set DIGITAL_CHAOS_BASELINE_MODE=openai
set DIGITAL_CHAOS_BASELINE_MODEL=gpt-4.1-mini
python baseline.py
```

Current reproducible heuristic baseline scores:
- easy: `0.8000`
- medium: `0.9750`
- hard: `0.7400`
- average: `0.8383`

## Required inference script (submission)

The project includes `inference.py` in the root directory, using OpenAI client with required variables:

- `API_BASE_URL` (LLM API base URL)
- `MODEL_NAME` (model id)
- `HF_TOKEN` (API key/token)

Optional:
- `ENV_BASE_URL` (defaults to `http://127.0.0.1:8000`)

Run:

```bash
set API_BASE_URL=<your_openai_compatible_url>
set MODEL_NAME=<your_model_name>
set HF_TOKEN=<your_token>
python inference.py
```

## Docker

```bash
docker build -t digital-chaos-env -f server/Dockerfile .
docker run -p 8000:8000 digital-chaos-env
```

## OpenEnv metadata

Configuration is in `openenv.yaml`.
