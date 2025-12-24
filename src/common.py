import numpy as np

# def rollout(env, agent, steps=200):
#     obs = env.reset()
#     log_probs, rewards = [], []

#     for _ in range(steps):
#         action, log_prob = agent.act(obs)
#         obs, reward, done, _ = env.step(action)

#         log_probs.append(log_prob)
#         rewards.append(reward)

#         if done:
#             break
#     log_probs.append(log_prob)
#     entropies.append(entropy)

#     return log_probs, rewards


def rollout(env, agent, steps=200):
    obs, _ = env.reset()

    obs_buf = []
    act_buf = []
    log_probs = []
    val_buf = []
    rew_buf = []
    done_buf = []

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
            break

    if done:
        next_val = 0
    else:
        _, _, next_val, _ = agent.act(obs)
        
    returns = []
    advantages = []
    gae = 0
    gamma = 0.99
    lam = 0.95
    
    values = val_buf + [next_val]
    
    for i in reversed(range(len(rew_buf))):
        delta = rew_buf[i] + gamma * values[i+1] * (1 - int(done_buf[i])) - values[i]
        gae = delta + gamma * lam * (1 - int(done_buf[i])) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])

    return obs_buf, act_buf, log_probs, returns, advantages



















# def rollout(env, agent, steps=200):
#     obs = env.reset()

#     obs_buf = []
#     actions = []
#     rewards = []
#     log_probs = []
#     values = []
#     entropies = []
#     dones = []

#     for _ in range(steps):
#         action, log_prob, value, entropy = agent.act(obs)

#         next_obs, reward, done, _ = env.step(action)

#         obs_buf.append(obs)
#         actions.append(action)
#         rewards.append(reward)
#         log_probs.append(log_prob)
#         values.append(value)
#         entropies.append(entropy)
#         dones.append(done)

#         obs = next_obs
#         if done:
#             break

#     returns, advantages = agent.compute_returns_advantages(
#         rewards, values, dones
#     )

#     return obs_buf, actions, log_probs, returns, advantages, entropies
