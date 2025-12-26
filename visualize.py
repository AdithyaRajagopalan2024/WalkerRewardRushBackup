import torch
import gym
import pickle
from src.agents.ppo_agent import PPOAgent

def visualize():
    env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
    agent = PPOAgent(env.observation_space.shape[0], env.action_space.shape[0])
    
    agent.policy.load_state_dict(torch.load("checkpoints/walker_final.pt"))
    with open("checkpoints/stats.pkl", "rb") as f:
        stats = pickle.load(f)
    
    obs, _ = env.reset()
    while True:
        action = agent.act(stats.normalize(obs), deterministic=True)[0]
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated: obs, _ = env.reset()

if __name__ == "__main__":
    visualize()