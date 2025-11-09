import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Golden ratio constants
phi = (1 + torch.sqrt(torch.tensor(5.0))) / 2          # ≈1.618
keep_prob = 1 / phi                                    # ≈0.618  → keep 61.8%
prune_prob = 1 - keep_prob                             # ≈0.382

# -------------------------------------------------
# 2. Toy data
torch.manual_seed(0)
x = torch.linspace(0, 2 * np.pi, 100).unsqueeze(1)
y = torch.sin(x) + 0.1 * torch.randn_like(x)

# -------------------------------------------------
# 3. Simple MLP (same as in your repo)
class PhiMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# 4. Loss that includes the η-drift regulariser
def phi_loss(model, x, y, drift):
    pred = model(x)
    mse  = nn.MSELoss()(pred, y)
    # η-drift penalty (g is a small coupling)
    g = 1e-4
    reg = g * sum(torch.mean(torch.abs(p)**(max(0.1, phi - drift)))
                  for p in model.parameters())
    return mse + reg, mse.item()

# -------------------------------------------------
# 5. Training hyper-parameters
epochs   = 200
drifts   = [0.0, 0.1, 0.2, 0.3, 0.4]
colors   = plt.cm.viridis(np.linspace(0, 1, len(drifts)))

# -------------------------------------------------
# 6. Plot setup
fig, ax = plt.subplots(figsize=(10, 6))

# -------------------------------------------------
# 7. MAIN LOOP
losses = {}
for i, drift in enumerate(drifts):
    model = PhiMLP()
    opt   = optim.Adam(model.parameters(), lr=0.01)
    loss_history = []

    for e in range(epochs):
        # ---- forward / loss ----
        total_loss, mse = phi_loss(model, x, y, drift)
        opt.zero_grad()
        total_loss.backward()
        opt.step()

        # ---- φ-sparsity pruning every 10 steps ----
        if (e + 1) % 10 == 0:
            with torch.no_grad():
                for p in model.parameters():
                    # hard threshold: kill entries smaller than 1/φ
                    p.data[torch.abs(p.data) < 1/phi] = 0.0

        # ---- golden-ratio dropout (soft mask) after each step ----
        with torch.no_grad():
            for p in model.parameters():
                mask = (torch.rand_like(p.data) > prune_prob).float()
                p.data = p.data * mask / keep_prob   # unbiased scaling

        loss_history.append(mse)

    # store results
    losses[drift] = loss_history
    ax.plot(range(epochs), loss_history,
            color=colors[i], label=f'Δη={drift}', linewidth=2)

# -------------------------------------------------
# 8. Finalise plot
ax.set_yscale('log')
ax.set_xlabel('Epochs')
ax.set_ylabel('MSE (log scale)')
ax.set_title('AGI training loss under η-drift\n'
             'with 1/φ sparsity (≈61.8 % kept)')
ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.savefig('agi_eta_drift.png', dpi=300)
plt.show()