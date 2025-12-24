# from src.envs.walker_env import SimpleWalkerEnv
from src.agents.ppo_agent import PPOAgent
from src.common import rollout
from src.eval import evaluate
import torch
import os
import gym
import numpy as np

# CHECK AGAIN IF BOOL8 IS VALID AND EXISTS
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def main():
    # env = SimpleWalkerEnv()
    env = gym.make("BipedalWalker-v3")

    agent = PPOAgent(obs_dim=env.observation_space.shape[0], act_dim=env.action_space.shape[0])

    for epoch in range(200):
        obs, actions, log_probs, returns, advantages = rollout(env, agent)
        agent.update(obs, actions, log_probs, returns, advantages)

        if epoch % 10 == 0:
            score = evaluate(env, agent, episodes=10)
            print(f"Epoch {epoch} | Eval Reward: {score:.2f}")

    os.makedirs("checkpoints", exist_ok=True)
    model_path = "checkpoints/walker.pt"
    torch.save(agent.policy.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()




# from src.envs.walker_env import SimpleWalkerEnv
# from src.agents.ppo_agent import PPOAgent
# from src.common import rollout
# from src.eval import evaluate
# import torch

# def main():
#     env = SimpleWalkerEnv()
#     agent = PPOAgent(obs_dim=2, act_dim=1)

#     for epoch in range(200):
#         obs, actions, log_probs, returns, advantages = rollout(env, agent)
#         agent.update(obs, actions, log_probs, returns, advantages)

#         if epoch % 10 == 0:
#             score = evaluate(env, agent, episodes=10)
#             print(f"Epoch {epoch} | Eval Reward: {score:.2f}")

#     torch.save(agent.policy.state_dict(), "checkpoints/walker.pt")

# if __name__ == "__main__":
#     main()

















# from src.envs.walker_env import SimpleWalkerEnv
# from src.agents.ppo_agent import PPOAgent
# from src.common import rollout
# from src.eval import evaluate
# import torch
# import os
# import gym

# def main():
#     env = SimpleWalkerEnv()
#     # env = gym.make("BipedalWalker-v3", render_mode="human")

#     agent = PPOAgent(obs_dim=2, act_dim=1)

#     for epoch in range(200):
#         obs, actions, log_probs, returns, advantages = rollout(env, agent)
#         agent.update(obs, actions, log_probs, returns, advantages)

#         if epoch % 10 == 0:
#             score = evaluate(env, agent, episodes=10)
#             print(f"Epoch {epoch} | Eval Reward: {score:.2f}")

#     os.makedirs("checkpoints", exist_ok=True)
#     model_path = "checkpoints/walker.pt"
#     torch.save(agent.policy.state_dict(), model_path)
#     print(f"\nModel saved to {model_path}")

# if __name__ == "__main__":
#     main()




# # from src.envs.walker_env import SimpleWalkerEnv
# # from src.agents.ppo_agent import PPOAgent
# # from src.common import rollout
# # from src.eval import evaluate
# # import torch

# # def main():
# #     env = SimpleWalkerEnv()
# #     agent = PPOAgent(obs_dim=2, act_dim=1)

# #     for epoch in range(200):
# #         obs, actions, log_probs, returns, advantages = rollout(env, agent)
# #         agent.update(obs, actions, log_probs, returns, advantages)

# #         if epoch % 10 == 0:
# #             score = evaluate(env, agent, episodes=10)
# #             print(f"Epoch {epoch} | Eval Reward: {score:.2f}")

# #     torch.save(agent.policy.state_dict(), "checkpoints/walker.pt")

# # if __name__ == "__main__":
# #     main()






