import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
if not hasattr(np, 'typeDict'):
    np.typeDict = {} 

from src.agents.ppo_agent import PPOAgent
from src.common import rollout, RunningStats
from src.eval import evaluate
import torch
import os
import gym
import pickle
from torch.utils.tensorboard import SummaryWriter


def main():
    env = gym.make("BipedalWalker-v3", hardcore=True)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = PPOAgent(obs_dim, act_dim, lr=1e-4, entropy_coef=0.02)
    stats = RunningStats(obs_dim)
    writer = SummaryWriter(log_dir="runs/walker_hardcore")

    for epoch in range(15000):
        obs_b, act_b, log_b, ret_b, adv_b = rollout(env, agent, stats, steps=2048)
        a_loss, c_loss = agent.update(obs_b, act_b, log_b, ret_b, adv_b)

        if epoch % 10 == 0:
            avg_reward = evaluate(env, agent, stats, episodes=5)
            print(f"Epoch {epoch:4d} | Reward: {avg_reward:7.2f} | LogStd: {agent.policy.log_std.mean().item():.2f}")
            writer.add_scalar("Reward/Eval", avg_reward, epoch)

        if epoch % 100 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(agent.policy.state_dict(), "checkpoints/walker_final.pt")
            with open("checkpoints/stats.pkl", "wb") as f:
                pickle.dump(stats, f)

if __name__ == "__main__":
    main()