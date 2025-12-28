import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque

# ======================================================
# Actor–Critic Network
# ======================================================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_size, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)

    def get_action_and_value(self, obs, action=None):
        x = self.actor(obs)
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        logprob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)

        return action, logprob, entropy, value

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)


# ======================================================
# Running Mean Std
# ======================================================
class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
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
        new_var = (m_a + m_b + delta**2 * self.count * batch_count / tot_count) / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count


# ======================================================
# PPO Agent
# ======================================================
class PPO:
    def __init__(
        self,
        env_name="Ant-v5",
        num_envs=4,
        num_steps=2048,
        total_timesteps=2_000_000,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=10,
        num_minibatches=32,
        normalize_obs=True,
        normalize_reward=True,
    ):
        self.env_name = env_name
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

        self.batch_size = num_envs * num_steps
        self.minibatch_size = self.batch_size // num_minibatches

        # ✅ Correct Gymnasium VectorEnv + episode stats
        self.envs = gym.vector.SyncVectorEnv(
            [
                lambda: gym.wrappers.RecordEpisodeStatistics(
                    gym.make(env_name)
                )
                for _ in range(num_envs)
            ]
        )

        obs_dim = np.prod(self.envs.single_observation_space.shape)
        act_dim = np.prod(self.envs.single_action_space.shape)

        self.agent = ActorCritic(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr, eps=1e-5)

        self.normalize_obs = normalize_obs
        self.normalize_reward = normalize_reward

        if normalize_obs:
            self.obs_rms = RunningMeanStd(self.envs.single_observation_space.shape)
        if normalize_reward:
            self.ret_rms = RunningMeanStd(())
            self.returns = np.zeros(num_envs)

        self.ep_returns = deque(maxlen=100)

    def _normalize_obs(self, obs):
        if self.normalize_obs:
            self.obs_rms.update(obs)
            obs = (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
            obs = np.clip(obs, -10, 10)
        return obs

    def _normalize_reward(self, reward, done):
        if self.normalize_reward:
            self.returns = self.returns * self.gamma + reward
            self.ret_rms.update(self.returns.reshape(-1))
            reward = reward / np.sqrt(self.ret_rms.var + 1e-8)
            self.returns[done] = 0.0
        return reward

    def train(self):
        obs, _ = self.envs.reset()
        obs = self._normalize_obs(obs)
        obs = torch.tensor(obs, dtype=torch.float32)

        num_updates = self.total_timesteps // self.batch_size

        print(f"\nTraining PPO on {self.env_name}")
        print(f"{'Upd':<5}{'AvgRew':<10}{'Ent':<8}{'KL':<10}{'Clip%':<8}{'EV':<8}")
        print("-" * 55)

        for update in range(1, num_updates + 1):
            obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []

            for _ in range(self.num_steps):
                with torch.no_grad():
                    action, logp, _, value = self.agent.get_action_and_value(obs)

                next_obs, reward, term, trunc, infos = self.envs.step(action.numpy())
                done = np.logical_or(term, trunc)

                # ✅ CORRECT Gymnasium vector episode logging
                if "episode" in infos:
                    for r in infos["episode"]["r"]:
                        if r is not None:
                            self.ep_returns.append(r)

                reward = self._normalize_reward(reward, done)

                obs_buf.append(obs)
                act_buf.append(action)
                logp_buf.append(logp)
                rew_buf.append(torch.tensor(reward))
                val_buf.append(value)
                done_buf.append(torch.tensor(done, dtype=torch.float32))

                obs = self._normalize_obs(next_obs)
                obs = torch.tensor(obs, dtype=torch.float32)

            with torch.no_grad():
                next_value = self.agent.get_value(obs)

            # ---------- GAE ----------
            advantages = []
            gae = 0
            for t in reversed(range(self.num_steps)):
                mask = 1.0 - done_buf[t]
                delta = rew_buf[t] + self.gamma * next_value * mask - val_buf[t]
                gae = delta + self.gamma * self.gae_lambda * mask * gae
                advantages.insert(0, gae)
                next_value = val_buf[t]

            advantages = torch.stack(advantages)
            returns = advantages + torch.stack(val_buf)

            # ---------- Flatten ----------
            obs_b = torch.cat(obs_buf)
            act_b = torch.cat(act_buf)
            logp_b = torch.cat(logp_buf)
            adv_b = advantages.flatten()
            ret_b = returns.flatten()
            val_b = torch.cat(val_buf)

            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            inds = np.arange(self.batch_size)
            clipfracs, approx_kls = [], []

            for _ in range(self.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    mb = inds[start:start + self.minibatch_size]

                    _, newlogp, entropy, newvalue = self.agent.get_action_and_value(
                        obs_b[mb], act_b[mb]
                    )

                    ratio = (newlogp - logp_b[mb]).exp()
                    approx_kl = (logp_b[mb] - newlogp).mean()
                    approx_kls.append(approx_kl.item())

                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                    )

                    pg_loss = torch.max(
                        -adv_b[mb] * ratio,
                        -adv_b[mb] * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef),
                    ).mean()

                    v_loss = 0.5 * (newvalue - ret_b[mb]).pow(2).mean()
                    loss = pg_loss + self.vf_coef * v_loss - self.ent_coef * entropy.mean()

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()

            ev = 1 - torch.var(ret_b - val_b) / torch.var(ret_b)
            avg_rew = np.mean(self.ep_returns) if self.ep_returns else 0.0

            print(
                f"{update:<5}{avg_rew:<10.1f}"
                f"{entropy.mean().item():<8.2f}"
                f"{np.mean(approx_kls):<10.4f}"
                f"{np.mean(clipfracs)*100:<8.2f}"
                f"{ev.item():<8.3f}"
            )

        self.envs.close()
        print("\n✅ Training complete")


# ======================================================
# Run
# ======================================================
if __name__ == "__main__":
    PPO().train()
