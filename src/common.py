import numpy as np

def rollout(env, agent, steps=2048):
    obs, _ = env.reset()
    obs_buf, act_buf, log_probs, val_buf, rew_buf, done_buf = [], [], [], [], [], []

    for _ in range(steps):
        action_clamped, log_prob, value, raw_action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action_clamped)
        done = terminated or truncated

        obs_buf.append(obs)
        act_buf.append(raw_action)
        log_probs.append(log_prob)
        val_buf.append(value)
        rew_buf.append(reward)
        done_buf.append(done)

        obs = next_obs
        if done:
            obs, _ = env.reset() # Continue collecting to fill the 2048 buffer

    _, _, last_val, _ = agent.act(obs)
    
    returns, advantages = [], []
    gae = 0
    gamma, lam = 0.99, 0.95
    values = val_buf + [last_val]
    
    for i in reversed(range(len(rew_buf))):
        # Only use next value if the episode didn't end
        mask = 1.0 - float(done_buf[i])
        delta = rew_buf[i] + gamma * values[i+1] * mask - values[i]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])

    return obs_buf, act_buf, log_probs, returns, advantages