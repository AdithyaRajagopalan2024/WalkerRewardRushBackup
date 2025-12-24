import numpy as np

def rollout(env, agent, steps=2048):
    obs, _ = env.reset()
    obs_buf, act_buf, log_probs, val_buf, rew_buf, done_buf = [], [], [], [], [], []

    for _ in range(steps):
        action_clamped, log_prob, value, raw_action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action_clamped)
        
        # 0: hull angle, 2: horizontal speed, 3: vertical speed
        # 4-13: LIDAR, 14-23: Leg joints/contacts
        
        # 1. Penalize "Hopping" (Vertical Velocity)
        reward -= 0.01 * abs(next_obs[3])
        
        # 2. Penalize "Jerky" Leg Movement (Torque Penalty)
        reward -= 0.001 * np.sum(np.square(raw_action))
        
        # 3. Upright Bonus
        if abs(next_obs[0]) < 0.1:
            reward += 0.005 
            
        # 4. Forward Velocity Bonus
        if next_obs[2] > 0.2:
            reward += 0.01

        done = terminated or truncated

        obs_buf.append(obs)
        act_buf.append(raw_action)
        log_probs.append(log_prob)
        val_buf.append(value)
        rew_buf.append(reward)
        done_buf.append(done)

        obs = next_obs
        if done:
            obs, _ = env.reset()

    _, _, last_val, _ = agent.act(obs)
    
    returns, advantages = [], []
    gae = 0
    gamma, lam = 0.98, 0.95
    values = val_buf + [last_val]
    
    for i in reversed(range(len(rew_buf))):
        mask = 1.0 - float(done_buf[i])
        delta = rew_buf[i] + gamma * values[i+1] * mask - values[i]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])

    return obs_buf, act_buf, log_probs, returns, advantages