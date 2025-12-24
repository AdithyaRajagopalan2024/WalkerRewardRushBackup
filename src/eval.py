def evaluate(env, agent, episodes=5):
    scores = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        scores.append(total_reward)

    return sum(scores) / len(scores)
