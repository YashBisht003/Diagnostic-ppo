PPO Implementation: Broken vs Fixed 
A side-by-side comparison of a deliberately broken PPO implementation and its corrected version for training on MuJoCo's Ant-v5 environment. This repository demonstrates how subtle bugs can cause silent failures while metrics appear normal.


# Install dependencies
pip install -r requirements.txt
Requirements
txtgymnasium[mujoco]>=0.29.0
torch>=2.0.0
numpy>=1.24.0
mujoco>=2.3.0
Running the Code
Broken Implementation:
bashpython broken_ppo.py
Fixed Implementation:
bashpython fixed_ppo.py
# The Broken Implementation
The 7 Deadly Bugs
The broken implementation contains these intentional bugs:

Wrong Advantage Normalization (Line 182)

python   # BUG: Dividing by tiny std inflates gradients
   advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

Corrupted KL Divergence (Line 207)

python   # BUG: Multiplies old log probs by 1.05, creating fake KL measurements
   ratio = torch.exp(log_probs - old_log_probs_t[batch_idx] * 1.05)

Inverted Policy Loss (Line 212)

python   # BUG: Should be -torch.min()! This maximizes loss instead of minimizing
   policy_loss = torch.min(surr1, surr2).mean()

No Value Clipping (Line 215)

python   # BUG: Value function clipping disabled - lets critic go wild
   value_loss = ((values - returns_t[batch_idx]) ** 2).mean()

Negative Entropy Bonus (Line 218)

python   # BUG: Discourages exploration instead of encouraging it
   entropy_loss = -entropy.mean()

Gradient Clipping Disabled (Line 226)

python   # BUG: Commented out - allows exploding gradients
   # nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

Incorrect Explained Variance (Line 236)

python   # BUG: Computed incorrectly - shows positive even when predictions are garbage
   explained_var = 1 - ((returns_t[batch_idx] - values).var() / (y_var + 1e-8))
What You'll Observe
Despite these bugs, the logs will show:

 Entropy decreasing (looks like convergence)
 Non-zero KL divergence (looks like learning)
 Clip fraction values (looks like PPO is working)
 Advantage statistics computed (looks professional)
 Positive explained variance (looks like critic works)
