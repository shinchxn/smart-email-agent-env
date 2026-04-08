"""
Smart Email Agent Environment - Inference Baseline.

A simple evaluation script demonstrating how to interact with 
the smart email environment using a basic heuristic.
"""

from client import SmartEmailEnv
from models import EmailAction



def run_heuristic_baseline():
    """Runs a simple heuristic baseline on the environment."""
    env = SmartEmailEnv(url="http://localhost:8000")
    
    print("[INFO] Starting heuristic baseline...")
    obs = env.reset()
    
    while not obs.done:
        print(f"\n[STEP] Processing Email ID: {obs.email_id}")
        
        # Simple heuristic: prioritize based on keywords
        text = obs.current_email_text.lower()
        
        category = "work"
        if any(w in text for w in ["invoice", "bank", "payment"]):
            category = "finance"
        elif any(w in text for w in ["party", "bbq", "meetup"]):
            category = "social"
        elif any(w in text for w in ["viagra", "lottery", "winner"]):
            category = "spam"
            
        priority = "medium"
        if any(w in text for w in ["urgent", "asap", "immediate"]):
            priority = "urgent"
        elif any(w in text for w in ["deadline", "critical"]):
            priority = "high"
            
        action = "reply"
        if category == "spam":
            action = "ignore"
        elif category == "social":
            action = "flag"
            
        # Formulate action
        act = EmailAction(
            email_id=obs.email_id,
            predicted_category=category,
            predicted_priority=priority,
            action_taken=action,
            optional_response="Understood, I will handle this." if action == "reply" else None
        )
        
        # Take step
        obs = env.step(act)
        print(f"[FEEDBACK] {obs.last_feedback}")
        print(f"[REWARD] Step Reward: {obs.cumulative_reward}")
        
    print(f"\n[DONE] Episode finished. Total Reward: {obs.cumulative_reward}")


if __name__ == "__main__":
    run_heuristic_baseline()
