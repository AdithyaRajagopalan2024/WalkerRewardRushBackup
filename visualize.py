import torch
import time
import sys
import gym
import numpy as np


if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

from src.agents.ppo_agent import PPOAgent

def visualize_rollout(agent, env):
    """Runs one episode and renders it."""
    obs, _ = env.reset()
    done = False

    while not done:
        env.render()
        time.sleep(0.02)
        time.sleep(0.02)
        action = agent.act(obs, deterministic=True)[0]

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    MODEL_PATH = "checkpoints/walker.pt"

    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
    agent = PPOAgent(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0])

    try:
        agent.policy.load_state_dict(torch.load(MODEL_PATH))
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'", file=sys.stderr)
        print("Please train the agent first by running: python train.py", file=sys.stderr)
        sys.exit(1)

    agent.policy.eval()

    print("Starting rollout visualization...")
    visualize_rollout(agent, env)


# import gym
# from stable_baselines3 import PPO

# env = gym.make("BipedalWalker-v3", render_mode="human")
# model = PPO.load("checkpoints/walker.pt")  
# observation, info = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample()

#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()

# env.close() # Close the visualization window after the loop
# Environment and model loading code using Stable Baselines3 only 