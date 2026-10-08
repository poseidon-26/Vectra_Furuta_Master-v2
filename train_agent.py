"""
train_agent.py — SAC training loop for the Furuta pendulum.

Key design choices:
  • Domain-randomised environment (DynamicFurutaEnv) with RK4 physics.
  • Standard SAC with clipped double-Q, fixed entropy temperature α.
  • log_prob uses the correct tanh-squashing correction from the SAC paper.
  • Periodic model checkpointing so a crash doesn't lose all progress.
"""

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from environment import DynamicFurutaEnv
from models import SACActor, SACCritic


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000):
        self.capacity = capacity
        self.buffer: list = []
        self.position: int = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Log-prob helper (tanh-squashing correction)
# ---------------------------------------------------------------------------

def _squashed_log_prob(
    dist: torch.distributions.Normal,
    z: torch.Tensor,
) -> torch.Tensor:
    """
    log π(a|s) = log μ(z|s) − Σ log(1 − tanh²(z_i) + ε)

    The 1e-6 epsilon prevents log(0) when |z| is large and tanh saturates.
    Summed over the action dimension, kept as [B, 1] for broadcasting with Q.
    """
    log_prob = dist.log_prob(z) - torch.log(1.0 - torch.tanh(z).pow(2) + 1e-6)
    return log_prob.sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_sim_to_real():
    print("[Trainer] Initiating Robust Physics Pipeline with Domain Randomisation...")

    env   = DynamicFurutaEnv()
    actor = SACActor()
    critic        = SACCritic()
    critic_target = copy.deepcopy(critic)

    actor_optimizer  = optim.Adam(actor.parameters(),  lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)

    gamma, tau, alpha = 0.99, 0.005, 0.2
    batch_size        = 256
    buffer            = ReplayBuffer()

    epochs              = 2000
    max_steps_per_epoch = 500
    exploration_steps   = 500
    save_interval       = 100       # checkpoint every N epochs
    total_steps         = 0

    for epoch in range(epochs):
        state        = env.reset()
        episode_reward = 0.0

        for _step in range(max_steps_per_epoch):
            state_arr = np.array(state, dtype=np.float32)

            # --- Action selection ---
            if total_steps < exploration_steps:
                action = np.random.uniform(-env.max_torque, env.max_torque, size=(1,))
            else:
                state_t = torch.FloatTensor(state_arr).unsqueeze(0)
                with torch.no_grad():
                    mean, log_std = actor(state_t)
                    dist          = torch.distributions.Normal(mean, log_std.exp())
                    z             = dist.rsample()
                    action        = (torch.tanh(z) * actor.max_action).cpu().numpy().flatten()

            next_state, reward, done = env.step(action)
            episode_reward += reward
            buffer.push(
                state_arr,
                action,
                reward,
                np.array(next_state, dtype=np.float32),
                float(done),
            )

            state        = next_state
            total_steps += 1

            # --- Update networks ---
            if len(buffer) > batch_size and total_steps > exploration_steps:
                b_s, b_a, b_r, b_ns, b_d = buffer.sample(batch_size)

                b_s  = torch.FloatTensor(b_s)
                b_a  = torch.FloatTensor(b_a)
                b_r  = torch.FloatTensor(b_r).unsqueeze(1)
                b_ns = torch.FloatTensor(b_ns)
                b_d  = torch.FloatTensor(b_d).unsqueeze(1)

                # ---- Critic update ----
                with torch.no_grad():
                    n_mean, n_log_std = actor(b_ns)
                    n_dist            = torch.distributions.Normal(n_mean, n_log_std.exp())
                    n_z               = n_dist.rsample()
                    n_action          = torch.tanh(n_z) * actor.max_action
                    n_log_prob        = _squashed_log_prob(n_dist, n_z)

                    t_q1, t_q2 = critic_target(b_ns, n_action)
                    target_q   = b_r + (1.0 - b_d) * gamma * (
                        torch.min(t_q1, t_q2) - alpha * n_log_prob
                    )

                c_q1, c_q2  = critic(b_s, b_a)
                critic_loss = F.mse_loss(c_q1, target_q) + F.mse_loss(c_q2, target_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # ---- Actor update ----
                mean, log_std = actor(b_s)
                dist          = torch.distributions.Normal(mean, log_std.exp())
                z             = dist.rsample()
                curr_a        = torch.tanh(z) * actor.max_action
                log_prob      = _squashed_log_prob(dist, z)

                q1_pi, q2_pi = critic(b_s, curr_a)
                actor_loss   = (alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # ---- Target network soft update ----
                for p, tp in zip(critic.parameters(), critic_target.parameters()):
                    tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

            if done:
                break

        # --- Logging & checkpointing ---
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch: {epoch+1:04d}/{epochs} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Steps: {total_steps}"
            )

        if (epoch + 1) % save_interval == 0:
            ckpt_path = f"furuta_actor_ep{epoch+1}.pth"
            torch.save(actor.state_dict(), ckpt_path)
            print(f"[Trainer] Checkpoint saved: {ckpt_path}")

    torch.save(actor.state_dict(), "furuta_actor.pth")
    print("[Trainer] Optimisation complete. Final payload saved: furuta_actor.pth")


if __name__ == "__main__":
    train_sim_to_real()