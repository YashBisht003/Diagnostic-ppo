import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()


        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )


        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))


        self.critic = nn.Linear(64, 1)

    def forward(self, state):
        features = self.shared(state)
        return features

    def get_action_and_value(self, state, action=None):
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_logstd)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(features).squeeze(-1)

        return action, log_prob, entropy, value


class BrokenPPO:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, n_epochs=10, batch_size=64):

        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.episode_rewards = deque(maxlen=100)

    def collect_rollouts(self, n_steps):
        states, actions, rewards, dones, values, log_probs = [], [], [], [], [], []

        state, _ = self.env.reset()
        episode_reward = 0

        for step in range(n_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.policy.get_action_and_value(state_tensor)

            action_np = action.squeeze(0).numpy()
            next_state, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())
            log_probs.append(log_prob.item())

            episode_reward += reward
            state = next_state

            if done:
                self.episode_rewards.append(episode_reward)
                state, _ = self.env.reset()
                episode_reward = 0


        advantages = []
        returns = []

        next_value = 0
        if not done:
            with torch.no_grad():
                next_value = self.policy.get_action_and_value(
                    torch.FloatTensor(state).unsqueeze(0)
                )[3].item()

        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - done
                next_value_i = next_value
            else:
                next_non_terminal = 1.0 - dones[i + 1]
                next_value_i = values[i + 1]

            delta = rewards[i] + self.gamma * next_value_i * next_non_terminal - values[i]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        return (np.array(states), np.array(actions), np.array(log_probs),
                np.array(values), np.array(advantages), np.array(returns))

    def update(self, states, actions, old_log_probs, old_values, advantages, returns):

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        old_log_probs_t = torch.FloatTensor(old_log_probs)
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        metrics = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'advantage_mean': [],
            'advantage_std': [],
            'explained_variance': []
        }

        for epoch in range(self.n_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                _, log_probs, entropy, values = self.policy.get_action_and_value(
                    states_t[batch_idx], actions_t[batch_idx]
                )


                ratio = torch.exp(log_probs - old_log_probs_t[batch_idx] * 1.05)

                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_t[batch_idx]


                policy_loss = torch.min(surr1, surr2).mean()


                value_loss = ((values - returns_t[batch_idx]) ** 2).mean()


                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()



                self.optimizer.step()


                with torch.no_grad():
                    kl_div = (old_log_probs_t[batch_idx] - log_probs).mean()
                    clip_fraction = ((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).float().mean()


                    y_var = returns_t[batch_idx].var()
                    explained_var = 1 - ((returns_t[batch_idx] - values).var() / (y_var + 1e-8))

                    metrics['policy_loss'].append(policy_loss.item())
                    metrics['value_loss'].append(value_loss.item())
                    metrics['entropy'].append(entropy.mean().item())
                    metrics['kl_divergence'].append(kl_div.item())
                    metrics['clip_fraction'].append(clip_fraction.item())
                    metrics['advantage_mean'].append(advantages_t[batch_idx].mean().item())
                    metrics['advantage_std'].append(advantages_t[batch_idx].std().item())
                    metrics['explained_variance'].append(explained_var.item())

        return {k: np.mean(v) for k, v in metrics.items()}

    def train(self, total_timesteps, n_steps=2048):
        n_updates = total_timesteps // n_steps

        print(f"Training Broken PPO on {self.env.spec.id}")
        print(f"Total timesteps: {total_timesteps}, Updates: {n_updates}\n")
        print(f"{'Update':<8} {'AvgRew':<10} {'Policy':<10} {'Value':<10} {'Entropy':<10} "
              f"{'KL':<10} {'Clip%':<10} {'AdvMean':<10} {'AdvStd':<10} {'ExplVar':<10}")
        print("-" * 108)

        for update in range(1, n_updates + 1):

            states, actions, log_probs, values, advantages, returns = self.collect_rollouts(n_steps)


            metrics = self.update(states, actions, log_probs, values, advantages, returns)


            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

            print(f"{update:<8} {avg_reward:<10.2f} {metrics['policy_loss']:<10.4f} "
                  f"{metrics['value_loss']:<10.4f} {metrics['entropy']:<10.4f} "
                  f"{metrics['kl_divergence']:<10.4f} {metrics['clip_fraction']:<10.4f} "
                  f"{metrics['advantage_mean']:<10.4f} {metrics['advantage_std']:<10.4f} "
                  f"{metrics['explained_variance']:<10.4f}")



if __name__ == "__main__":
    env = gym.make("Ant-v4")

    agent = BrokenPPO(
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=10,
        batch_size=64
    )

    agent.train(total_timesteps=100000, n_steps=2048)

    env.close()
