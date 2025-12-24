import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(64, act_dim)
        self.critic = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.shared(x)
        return self.actor_mean(x), self.critic(x)


class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        
        self.policy = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, obs, deterministic=False):
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            mean, value = self.policy(obs_t)
            log_std = torch.clamp(self.policy.log_std, -2, 1)
            std = torch.exp(log_std)
            
            dist = torch.distributions.Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.sample()
                
            log_prob = dist.log_prob(action).sum()
            
            action_clamped = torch.clamp(action, -1.0, 1.0)
        
        return action_clamped.numpy(), log_prob.item(), value.item(), action.numpy()

    def update(self, obs, actions, log_probs_old, returns, advantages):
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32)
        returns = torch.tensor(np.array(returns), dtype=torch.float32)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32)
        
        for _ in range(10):
            mean, values = self.policy(obs)
            log_std = torch.clamp(self.policy.log_std, -2, 1)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            
            log_probs = dist.log_prob(actions).sum(dim=1)
            entropy = dist.entropy().sum(dim=1).mean()
            
            ratio = torch.exp(log_probs - log_probs_old)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            loss = actor_loss + critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()























# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# class ActorCritic(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super().__init__()
#         self.shared = nn.Sequential(
#             nn.Linear(obs_dim, 64),
#             nn.Tanh(),
#             nn.Linear(64, 64),
#             nn.Tanh(),
#         )

#         self.actor_mean = nn.Linear(64, act_dim)
#         self.log_std = nn.Parameter(torch.zeros(act_dim))
#         self.critic = nn.Linear(64, 1)

#     def forward(self, x):
#         x = self.shared(x)
#         return self.actor_mean(x), self.critic(x)


# class PPOAgent:
#     def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_eps=0.2):
#         self.gamma = gamma
#         self.clip_eps = clip_eps

#         self.model = ActorCritic(obs_dim, act_dim)
#         self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

#     def act(self, obs, deterministic=False):
#         obs_t = torch.tensor(obs, dtype=torch.float32)
#         mean, value = self.model(obs_t)

#         log_std = torch.clamp(self.model.log_std, -2, 1)
#         std = torch.exp(log_std)

#         dist = torch.distributions.Normal(mean, std)

#         if deterministic:
#             action = mean
#         else:
#             action = dist.sample()

#         log_prob = dist.log_prob(action).sum()
#         entropy = dist.entropy().sum()

#         action = torch.clamp(action, -1.0, 1.0)

#         return (
#             action.detach().numpy(),
#             log_prob.detach(),
#             value.detach(),
#             entropy.detach(),
#         )

#     def compute_returns_advantages(self, rewards, values, dones):
#         returns = []
#         advs = []

#         G = 0
#         for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
#             G = r + self.gamma * G * (1 - d)
#             returns.insert(0, G)

#         returns = torch.tensor(returns, dtype=torch.float32)
#         values = torch.tensor(values, dtype=torch.float32).squeeze()
#         advs = returns - values
#         advs = (advs - advs.mean()) / (advs.std() + 1e-8)

#         return returns, advs

#     def update(self, obs, actions, log_probs_old, returns, advantages, entropies):
#         obs = torch.tensor(obs, dtype=torch.float32)
#         actions = torch.tensor(actions, dtype=torch.float32)
#         log_probs_old = torch.stack(log_probs_old)

#         mean, values = self.model(obs)
#         log_std = torch.clamp(self.model.log_std, -2, 1)
#         std = torch.exp(log_std)

#         dist = torch.distributions.Normal(mean, std)
#         log_probs = dist.log_prob(actions).sum(axis=1)
#         entropy = torch.stack(entropies).mean()

#         ratio = torch.exp(log_probs - log_probs_old)

#         surr1 = ratio * advantages
#         surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

#         actor_loss = -torch.min(surr1, surr2).mean()
#         critic_loss = (returns - values.squeeze()).pow(2).mean()

#         loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
