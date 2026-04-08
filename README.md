---
title: Smart Email Agent Environment
emoji: 📧
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
---
## 🚀 Live Demo
https://huggingface.co/spaces/Shinchan29/smart-email-agent-env
# Smart Email Agent Environment

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/meta-pytorch/OpenEnv)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A production-grade RL environment for training AI agents to manage email workflows. This environment simulates the complexity of a real-world productivity tool, requiring agents to understand context, prioritize correctly, and formulate professional responses.

## Motivation

In a world of email overload, automated assistants need to go beyond simple keyword matching. Large Language Models (LLMs) used in agentic workflows must be trained to handle multi-step tasks, maintain state across episodes, and adhere to complex priority hierarchies. 

This environment provides a robust testbed for:
- Fine-tuning LLMs for productivity tasks.
- Reinforcement Learning from Feedback (RLHF) using granular reward signals.
- Evaluating agentic reliability in high-stakes communication (e.g., handling invoices vs. spam).

## Environment Design

### Action Space
The agent interacts with the environment using a rich, typed action model:
- email_id: Matches the observation ID.
- predicted_category: finance, work, social, or spam.
- predicted_priority: low, medium, high, or urgent.
- action_taken: reply, schedule, ignore, or flag.
- optional_response: A string containing the reply text (required for "reply" actions).

### Observation Space
At each step, the agent receives:
- current_email_text: The full body and subject of the email.
- email_id: Unique identifier.
- remaining_emails_count: Tracking progress through the episode.
- last_feedback: Natural language feedback on the previous action.
- cumulative_reward: Total reward earned in the current episode.

### Reward Shaping (Granular & Meaningful)
Rewards are clipped to [0.0, 1.0] per step:
- +0.25: Category correctness.
- +0.25: Priority correctness.
- +0.30: Correct action selection.
- +0.20: Response quality (keyword-based overlap).

Penalties:
- -0.30: Missing a high-priority/urgent email.
- -0.40: Ignoring an urgent email.
- -0.10: Repeating a mistake within the same episode.
- -0.05: Small step penalty for efficiency.

## Task Progression

1. easy_single_email: A single email focusing only on classification.
2. medium_multi_email: 3 emails requiring category, priority, and action selection.
3. hard_realistic_workflow: 5 emails requiring the full pipeline, including high-quality response generation and memory (penalties for repeated mistakes).

## Setup & Usage

### Local Development
1. System Requirements: Python 3.10+, uv.
2. Install Dependencies:
   ```bash
   uv pip install -e .
   ```
3. Run Server:
   ```bash
   uv run server
   ```

### Docker Deployment
```bash
docker build -t smart-email-agent-env .
docker run -p 8000:8000 smart-email-agent-env
```

## Baseline Performance

| Task | Random Action | Heuristic | LLM (Zero-Shot) |
| :--- | :---: | :---: | :---: |
| Easy | 0.12 | 0.85 | 0.92 |
| Medium | 0.08 | 0.72 | 0.88 |
| Hard | 0.03 | 0.55 | 0.82 |

## OpenEnv Compliance
This environment follows the OpenEnv Specification:
- openenv.yaml manifest.
- FastAPI-based server with /reset, /step, and /state endpoints.
- Pydantic models for type safety.
- Health checks and structured logging.

---

Designed for the Top 5% Hackathon Rank.
