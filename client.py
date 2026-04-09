"""
Smart Email Agent Environment - OpenEnv Client.

Implements the mandatory OpenEnv client interface to connect 
the environment server with LLM-based agents.
"""

from openenv_core import EnvClient
from models import EmailAction, EmailObservation


class SmartEmailEnv(EnvClient):
    """Client for the Smart Email Agent environment."""

    def _parse_result(self, raw_result: dict) -> EmailObservation:
        return EmailObservation(**raw_result)

    def _parse_state(self, raw_state: dict) -> dict:
        return raw_state

    def _step_payload(self, action: EmailAction) -> dict:
        return action.model_dump()

    def reset(self, task_name: str = "easy_single_email") -> EmailObservation:
        """Reset the environment and return the first observation."""
        response = self.session.post(f"{self.url}/reset", params={"task_name": task_name})
        response.raise_for_status()
        return self._parse_result(response.json())

    def step(self, action: EmailAction) -> EmailObservation:
        """Take a step in the environment."""
        response = self.session.post(
            f"{self.url}/step", json=self._step_payload(action)
        )
        response.raise_for_status()
        return self._parse_result(response.json())

    def state(self) -> dict:
        """Return the current internal state of the environment."""
        response = self.session.get(f"{self.url}/state")
        response.raise_for_status()
        return self._parse_state(response.json())
