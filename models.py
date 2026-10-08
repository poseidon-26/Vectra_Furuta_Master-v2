import torch
import torch.nn as nn


class SACActor(nn.Module):
    """
    Stochastic actor for SAC.  Outputs a Gaussian distribution over actions;
    actions are squashed through tanh and scaled to [-max_action, max_action].

    forward()   → (mean, log_std)   used during training.
    get_action()→ deterministic tanh(mean)*max_action  used during inference.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 1, max_action: float = 10.0):
        super().__init__()
        self.max_action = max_action

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),       nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, state: torch.Tensor):
        """Returns (mean, log_std) — both shape [B, action_dim]."""
        x       = self.net(state)
        mean    = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        return mean, log_std

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Deterministic inference path: tanh(mean) * max_action.
        Using mean (no sampling) is the standard SAC evaluation policy.
        """
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.max_action


class SACCritic(nn.Module):
    """
    Twin Q-network (clipped double-Q) for SAC.
    Inputs:  state [B, state_dim],  action [B, action_dim]
    Outputs: Q1 [B, 1],  Q2 [B, 1]
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 1):
        super().__init__()

        def _mlp():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
                nn.Linear(256, 256),                    nn.ReLU(),
                nn.Linear(256, 1),
            )

        self.q1_net = _mlp()
        self.q2_net = _mlp()

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        return self.q1_net(sa), self.q2_net(sa)