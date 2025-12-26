import numpy as np
import gym

class RunningStats:
    def __init__(self, shape):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

def rollout(env, agent, stats, steps=2048):
    obs, _ = env.reset()
    obs_buf, act_buf, log_probs, val_buf, rew_buf, done_buf = [], [], [], [], [], []

    for _ in range(steps):
        norm_obs = stats.normalize(obs)
        action_clamped, log_prob, value, raw_action = agent.act(norm_obs)
        
        total_reward = 0
        for _ in range(2):
            next_obs, reward, terminated, truncated, _ = env.step(action_clamped)
            total_reward += reward
            if terminated or truncated: break
        
        if abs(next_obs[2]) < 0.05:
            total_reward -= 0.01 

        done = terminated or truncated
        obs_buf.append(norm_obs)
        act_buf.append(raw_action)
        log_probs.append(log_prob)
        val_buf.append(value)
        rew_buf.append(total_reward)
        done_buf.append(done)

        obs = next_obs
        if done: obs, _ = env.reset()

    stats.update(np.array(obs_buf))
    _, _, last_val, _ = agent.act(stats.normalize(obs))
    
    returns, advantages = [], []
    gae, gamma, lam = 0, 0.99, 0.95
    values = val_buf + [last_val]
    for i in reversed(range(len(rew_buf))):
        mask = 1.0 - float(done_buf[i])
        delta = rew_buf[i] + gamma * values[i+1] * mask - values[i]
        gae = delta + gamma * lam * mask * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[i])

    return obs_buf, act_buf, log_probs, returns, advantages





