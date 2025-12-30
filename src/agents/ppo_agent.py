import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal initialization for stability in deep RL"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        """Returns the mean action and the state value"""
        return torch.tanh(self.actor(x)), self.critic(x)

class PPOAgent:
    def __init__(self, obs_dim, act_dim, lr=3e-4, gamma=0.99, clip_eps=0.2, entropy_coef=0.02):
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.batch_size = 128
        self.n_epochs = 10
        
        self.policy = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

    def act(self, obs, deterministic=False):
        """Standard action selection logic"""
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, value = self.policy(obs_t)
            std = torch.exp(torch.clamp(self.policy.log_std, -3, 0.5))
            dist = torch.distributions.Normal(mean, std)
            
            action = mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action_clamped = torch.clamp(action, -1.0, 1.0)
        
        return action_clamped.squeeze(0).numpy(), log_prob.item(), value.item(), action.squeeze(0).numpy()

    def update(self, obs, actions, log_probs_old, returns, advantages):
        """PPO Update loop with advantage normalization"""
        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32)
        returns = torch.tensor(np.array(returns), dtype=torch.float32)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        update_counts = 0
        
        inds = np.arange(obs.shape[0])
        for _ in range(self.n_epochs):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], self.batch_size):
                end = start + self.batch_size
                mb_inds = inds[start:end]
                
                new_mean, new_values = self.policy(obs[mb_inds])
                std = torch.exp(torch.clamp(self.policy.log_std, -3, 0.5))
                dist = torch.distributions.Normal(new_mean, std)
                
                new_log_probs = dist.log_prob(actions[mb_inds]).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - log_probs_old[mb_inds])
                surr1 = ratio * advantages[mb_inds]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[mb_inds]
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (returns[mb_inds] - new_values.squeeze()).pow(2).mean()
                
                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                update_counts += 1
                
        return total_actor_loss / update_counts, total_critic_loss / update_counts