def evaluate(env, agent, stats, episodes=5):
    scores = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, total_reward = False, 0
        while not done:
            norm_obs = stats.normalize(obs)
            action = agent.act(norm_obs, deterministic=True)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        scores.append(total_reward)
    return sum(scores) / len(scores)