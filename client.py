"""
Smart Email Agent Environment - OpenEnv Client.

Implements the mandatory OpenEnv client interface to connect 
the environment server with LLM-based agents.
"""

from openenv_core import EnvClient
from models import EmailAction, EmailObservation


class SmartEmailEnv(EnvClient):
    """Client for the Smart Email Agent environment."""

    def reset(self) -> EmailObservation:
        """Reset the environment and return the first observation."""
        response = self.session.post(f"{self.url}/reset")
        response.raise_for_status()
        return EmailObservation(**response.json())

    def step(self, action: EmailAction) -> EmailObservation:
        """Take a step in the environment."""
        response = self.session.post(
            f"{self.url}/step", json=action.model_dump()
        )
        response.raise_for_status()
        return EmailObservation(**response.json())

    def state(self) -> dict:
        """Return the current internal state of the environment."""
        response = self.session.get(f"{self.url}/state")
        response.raise_for_status()
        return response.json()
