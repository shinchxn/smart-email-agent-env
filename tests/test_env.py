import pytest
from server.env import SmartEmailAgentEnv
from models import EmailAction

@pytest.fixture
def env():
    return SmartEmailAgentEnv()

def test_reset_default(env):
    obs = env.reset()
    assert obs.email_id == 1
    assert "medium_multi_email" in obs.last_feedback
    assert obs.remaining_emails_count == 2
    assert not obs.done

def test_reset_easy(env):
    obs = env.reset(task_name="easy_single_email")
    assert obs.email_id == 1
    assert obs.remaining_emails_count == 0
    assert not obs.done

def test_step_correct(env):
    obs = env.reset(task_name="easy_single_email")
    action = EmailAction(
        email_id=1,
        predicted_category="finance",
        predicted_priority="high",
        action_taken="reply",
        optional_response="Sure, I will handle the invoice."
    )
    new_obs = env.step(action)
    assert new_obs.done
    assert new_obs.cumulative_reward > 0.5  # Should get high reward for correct action
    assert "[OK] Category correct." in new_obs.last_feedback

def test_step_incorrect_category(env):
    obs = env.reset(task_name="easy_single_email")
    action = EmailAction(
        email_id=1,
        predicted_category="social",  # Correct is finance
        predicted_priority="high",
        action_taken="reply",
        optional_response="Sure, I will handle the invoice."
    )
    new_obs = env.step(action)
    assert "[ERR] Category mismatch." in new_obs.last_feedback
    assert new_obs.cumulative_reward < 0.8  # Penalty for mismatch

def test_invalid_email_id(env):
    env.reset(task_name="easy_single_email")
    action = EmailAction(
        email_id=99,  # Invalid ID
        predicted_category="finance",
        predicted_priority="high",
        action_taken="reply"
    )
    with pytest.raises(ValueError, match="does not match current email"):
        env.step(action)

def test_episode_termination(env):
    env.reset(task_name="medium_multi_email")
    # Step 1
    env.step(EmailAction(email_id=1, predicted_category="finance", predicted_priority="high", action_taken="reply", optional_response="ok"))
    # Step 2
    env.step(EmailAction(email_id=2, predicted_category="social", predicted_priority="low", action_taken="reply", optional_response="ok"))
    # Step 3
    obs = env.step(EmailAction(email_id=3, predicted_category="spam", predicted_priority="low", action_taken="ignore"))
    
    assert obs.done
    assert obs.email_id == -1
    assert "EPISODE DONE" in obs.current_email_text
