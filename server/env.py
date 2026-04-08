"""
Smart Email Agent Environment - Core Logic.

Implements the reinforcement learning logic, email task progression,
and reward calculations.
"""

import uuid
from typing import Dict, List, Any
from models import EmailAction, EmailObservation, EmailState


class SmartEmailAgentEnv:
    """Core RL environment logic."""

    def __init__(self):
        # Email pool defined by task difficulty
        self.task_emails = {
            "easy_single_email": [
                {
                    "id": 1,
                    "text": "Subject: Invoice #4502 from Cloud Services. Hi, please find the attached invoice for last month's usage.",
                    "category": "finance",
                    "priority": "high",
                    "correct_action": "reply",
                }
            ],
            "medium_multi_email": [
                {
                    "id": 1,
                    "text": "Subject: Invoice #4502 from Cloud Services. Hi, please find the attached invoice for last month's usage.",
                    "category": "finance",
                    "priority": "high",
                    "correct_action": "reply",
                },
                {
                    "id": 2,
                    "text": "Subject: BBQ this Saturday? Hey! Just wanted to check if you're coming to the BBQ. Let me know!",
                    "category": "social",
                    "priority": "low",
                    "correct_action": "reply",
                },
                {
                    "id": 3,
                    "text": "Subject: You won $1,000,000!! Click here to claim your prize now! Limited time offer.",
                    "category": "spam",
                    "priority": "low",
                    "correct_action": "ignore",
                },
            ],
            "hard_realistic_workflow": [
                {
                    "id": 1,
                    "text": "Subject: Invoice #4502 from Cloud Services. Hi, please find the attached invoice for last month's usage.",
                    "category": "finance",
                    "priority": "high",
                    "correct_action": "reply",
                },
                {
                    "id": 2,
                    "text": "Subject: URGENT: Server Down. Our production server is not responding. Immediate attention required.",
                    "category": "work",
                    "priority": "urgent",
                    "correct_action": "flag",
                },
                {
                    "id": 3,
                    "text": "Subject: BBQ this Saturday? Hey! Just wanted to check if you're coming to the BBQ. Let me know!",
                    "category": "social",
                    "priority": "low",
                    "correct_action": "reply",
                },
                {
                    "id": 4,
                    "text": "Subject: Project Update. The new feature set is ready for review. Please check the branch.",
                    "category": "work",
                    "priority": "medium",
                    "correct_action": "reply",
                },
                {
                    "id": 5,
                    "text": "Subject: You won $1,000,000!! Click here to claim your prize now! Limited time offer.",
                    "category": "spam",
                    "priority": "low",
                    "correct_action": "ignore",
                },
            ]
        }
        self.state_store: Dict[str, EmailState] = {}
        self._current_episode_id: str = str(uuid.uuid4())
        self._active_emails: List[Dict] = []

    def reset(self, task_name: str = "medium_multi_email") -> EmailObservation:
        """Reset the environment state for a specific task."""
        if task_name not in self.task_emails:
            raise ValueError(f"Invalid task_name: {task_name}")

        self._current_episode_id = str(uuid.uuid4())
        self._active_emails = self.task_emails[task_name]
        
        new_state = EmailState(
            episode_id=self._current_episode_id,
            task_name=task_name
        )
        self.state_store[self._current_episode_id] = new_state
        
        first_email = self._active_emails[0]
        return EmailObservation(
            current_email_text=first_email["text"],
            email_id=first_email["id"],
            remaining_emails_count=len(self._active_emails) - 1,
            last_feedback=f"Environment reset for task: {task_name}",
            cumulative_reward=0.0,
            done=False
        )

    def step(self, action: EmailAction) -> EmailObservation:
        """Process an action and return the next observation."""
        state = self.state_store.get(self._current_episode_id)
        if not state:
            raise ValueError("Episode not initialized. Call reset() first.")
            
        current_email = self._active_emails[state.current_email_index]
        if action.email_id != current_email["id"]:
            raise ValueError(f"Action email_id {action.email_id} does not match current email {current_email['id']}.")

        # Calculate reward
        reward = 0.0
        feedback_parts = []
        
        # 1. Category check (+0.25)
        if action.predicted_category == current_email["category"]:
            reward += 0.25
            feedback_parts.append("[OK] Category correct.")
        else:
            feedback_parts.append("[ERR] Category mismatch.")
            
        # 2. Priority check (+0.25)
        if action.predicted_priority == current_email["priority"]:
            reward += 0.25
            feedback_parts.append("[OK] Priority correct.")
        else:
            feedback_parts.append("[ERR] Priority mismatch.")
            
        # 3. Action check (+0.30)
        if action.action_taken == current_email["correct_action"]:
            reward += 0.30
            feedback_parts.append("[OK] Action correct.")
        else:
            reward -= 0.10
            feedback_parts.append("[ERR] Action incorrect.")
            
        # 4. Response quality check (+0.20)
        if action.action_taken == "reply":
            if action.optional_response and len(action.optional_response) > 5:
                reward += 0.20
                feedback_parts.append("[OK] Response quality met.")
            else:
                feedback_parts.append("[ERR] Response too short.")

        # Penalize repeated mistakes
        mistake_key = f"{action.predicted_category}_{action.predicted_priority}"
        if mistake_key in state.mistakes:
            reward -= 0.10
            feedback_parts.append("[PENALTY] Repeated mistake.")
        else:
            if "ERR" in "".join(feedback_parts):
                state.mistakes.append(mistake_key)

        state.cumulative_reward += reward
        state.current_email_index += 1
        state.step_count += 1
        
        done = state.current_email_index >= len(self._active_emails)
        
        if done:
            return EmailObservation(
                current_email_text="EPISODE DONE",
                email_id=-1,
                remaining_emails_count=0,
                last_feedback=" | ".join(feedback_parts) + " [DONE]",
                cumulative_reward=state.cumulative_reward,
                done=True
            )
        
        next_email = self._active_emails[state.current_email_index]
        return EmailObservation(
            current_email_text=next_email["text"],
            email_id=next_email["id"],
            remaining_emails_count=len(self._active_emails) - state.current_email_index - 1,
            last_feedback=" | ".join(feedback_parts),
            cumulative_reward=state.cumulative_reward,
            done=False
        )

    def state(self) -> Dict[str, Any]:
        """Return raw state for debugging."""
        state = self.state_store.get(self._current_episode_id)
        return state.model_dump() if state else {}
