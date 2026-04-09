"""
Smart Email Agent Environment - Evaluation Baseline.

Demonstrates a full LLM-based agent loop using the OpenAI client
complying with OpenEnv structured logging requirements.
"""

import os
import json
import time
from typing import List

from openai import OpenAI
from client import SmartEmailEnv
from models import EmailAction, EmailObservation

# ---------------------------------------------------------------------------
# Configuration (OpenEnv Checklist Compliant)
# ---------------------------------------------------------------------------
# API_BASE_URL: Points to the LLM inference endpoint (e.g. HF Router)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/hf-inference/v1")
# MODEL_NAME: The model to use for inference
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
# HF_TOKEN: Required for authentication (no default)
HF_TOKEN = os.getenv("HF_TOKEN")

# ENV_URL: Points to the environment server
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

# Initialize OpenAI client
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")


def get_llm_action(obs: EmailObservation) -> EmailAction:
    """Uses the LLM to decide on an action based on the observation."""
    system_prompt = (
        "You are an expert email assistant. Analyze the incoming email and output a strictly formatted JSON response. "
        "The JSON MUST include 'email_id', 'predicted_category', 'predicted_priority', 'action_taken', and optionally 'optional_response'.\n\n"
        "Categories: finance, work, social, spam\n"
        "Priorities: low, medium, high, urgent\n"
        "Actions: reply, schedule, ignore, flag\n"
    )
    
    user_prompt = f"Observation: {obs.model_dump_json()}"
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"}
        )
        content = completion.choices[0].message.content
        data = json.loads(content)
        # Ensure ID matches observation
        data["email_id"] = obs.email_id
        return EmailAction(**data)
    except Exception as e:
        # Fallback to a safe action if LLM fails
        return EmailAction(
            email_id=obs.email_id,
            predicted_category="work",
            predicted_priority="medium",
            action_taken="ignore"
        )


def run_evaluation(task_name: str):
    """Runs a structured evaluation for a specific task."""
    env = SmartEmailEnv(base_url=ENV_URL)
    
    # [START] Log
    print(f"[START] task={task_name} env=smart_email_agent_env model={MODEL_NAME}")
    
    try:
        obs = env.reset(task_name=task_name)
    except Exception as e:
        print(f"[ERROR] Failed to reset env: {e}")
        return

    step_n = 1
    total_steps = 0
    all_rewards = []
    
    while not obs.done:
        # Get action from LLM
        start_time = time.time()
        action = get_llm_action(obs)
        
        # Take step
        try:
            prev_reward = obs.cumulative_reward
            obs = env.step(action)
            step_reward = obs.cumulative_reward - prev_reward
            all_rewards.append(step_reward)
            
            # [STEP] Log
            act_str = f"cat={action.predicted_category}|pri={action.predicted_priority}|act={action.action_taken}"
            print(f"[STEP] step={step_n} action={act_str} reward={step_reward:.2f} done={str(obs.done).lower()} error=null")
            
            step_n += 1
            total_steps += 1
        except Exception as e:
            print(f"[STEP] step={step_n} action=error reward=0.00 done=true error={str(e)}")
            break
            
    # [END] Log
    success = obs.done and obs.email_id == -1
    reward_str = ",".join([f"{r:.2f}" for r in all_rewards])
    print(f"[END] success={str(success).lower()} steps={total_steps} score={obs.cumulative_reward:.2f} rewards={reward_str}")


if __name__ == "__main__":
    if not HF_TOKEN:
        print("[WARNING] HF_TOKEN is not set. Inference will likely fail unless using a local router.")
        
    tasks = [
        "easy_single_email",
        "medium_multi_email",
        "hard_realistic_workflow"
    ]
    
    for task in tasks:
        run_evaluation(task)
        print("-" * 40)
