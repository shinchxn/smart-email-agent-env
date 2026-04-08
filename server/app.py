"""
Smart Email Agent Environment - FastAPI Server.

Main entry point for the FastAPI server that hosts the 
RL environment endpoints.
"""

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from .env import SmartEmailAgentEnv
from models import EmailAction, EmailObservation


app = FastAPI(title="Smart Email Agent Environment")

# Global environment instance (in-memory for this example)
_env_instance = SmartEmailAgentEnv()


@app.get("/health")
def health_check():
    """Health check endpoint for OpenEnv verification."""
    return {"status": "healthy", "env": "smart_email_agent_env"}


@app.post("/reset", response_model=EmailObservation)
def reset(
    task_name: str = Query(
        "medium_multi_email", 
        description="Task difficulty preset: easy_single_email, medium_multi_email, hard_realistic_workflow"
    )
):
    """Reset the environment state for a specific task."""
    try:
        obs = _env_instance.reset(task_name=task_name)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=EmailObservation)
def step(action: EmailAction):
    """Execute a step based on the provided action."""
    try:
        obs = _env_instance.step(action)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def get_state():
    """Get the current internal state (for observability)."""
    return _env_instance.state()


def main():
    """Main entry point for the server script."""
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
