"""
Smart Email Agent Environment - Pydantic Models.

Defines the typed Action, Observation, and State dataclasses
for the smart email agent RL environment.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class EmailAction(BaseModel):
    """Action taken by the agent for a single email step."""

    email_id: int = Field(
        ..., description="ID of the email being processed."
    )
    predicted_category: str = Field(
        ...,
        description="Predicted category: finance, work, social, or spam.",
    )
    predicted_priority: str = Field(
        ...,
        description="Predicted priority: low, medium, high, or urgent.",
    )
    action_taken: str = Field(
        ...,
        description="Action to take: reply, schedule, ignore, or flag.",
    )
    optional_response: Optional[str] = Field(
        default=None,
        description="Short reply text when action_taken is 'reply'.",
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class EmailObservation(BaseModel):
    """Observation returned to the agent after each step."""

    current_email_text: str = Field(
        ..., description="Body text of the current email to process."
    )
    email_id: int = Field(
        ..., description="Unique ID of the current email."
    )
    remaining_emails_count: int = Field(
        ..., description="Number of emails left to process after this one."
    )
    last_feedback: str = Field(
        default="",
        description="Human-readable feedback string from the last step.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total accumulated reward so far in this episode.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )


# ---------------------------------------------------------------------------
# State (internal episode metadata)
# ---------------------------------------------------------------------------

class EmailState(BaseModel):
    """Internal state of the environment for a single episode."""

    episode_id: str = Field(
        ..., description="Unique identifier for the episode."
    )
    step_count: int = Field(
        default=0, description="Number of steps taken so far."
    )
    current_email_index: int = Field(
        default=0, description="Index of the next email to serve."
    )
    processed_emails: List[int] = Field(
        default_factory=list,
        description="IDs of emails already processed.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total accumulated reward.",
    )
    mistakes: List[str] = Field(
        default_factory=list,
        description="Track repeated mistake categories for penalty.",
    )
    task_name: str = Field(
        default="medium_multi_email",
        description="Active task name.",
    )
