import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Toy AGI sim: φ-tuned MLP regressor with η-drift in loss
phi = (1 + np.sqrt(5)) / 2
eta_nominal = 1 - 1 / phi  # ≈0.381966

# Data: Simple regression y = sin(x) + noise, x in [0, 2π]
x = torch.linspace(0, 2 * np.pi, 100).unsqueeze(1)
y = torch.sin(x) + 0.1 * torch.randn_like(x)

# MLP: 1->32->1
class PhiMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# φ-Loss: MSE + g/4! ||θ||^φ (η-dressed: exponent phi - eta_drift)
def phi_loss(model, x, y, eta_drift=0):
    pred = model(x)
    mse = nn.MSELoss()(pred, y)
    # η-drift reg: g * mean |θ|^{max(0.1, phi - drift)} for stability
    g = 1e-4  # Small for balance
    reg = 0
    for p in model.parameters():
        theta = p.view(-1)
        exp = max(0.1, phi - eta_drift)
        reg += g * torch.mean(torch.abs(theta) ** exp)
    return mse + reg, mse.item()

# Sim: Train for 200 epochs, vary eta_drift, plot loss
epochs = 200
drifts = [0.0, 0.1, 0.2, 0.3, 0.4]  # Nominal to >0.382
losses = {d: [] for d in drifts}
colors = plt.cm.viridis(np.linspace(0, 1, len(drifts)))

fig, ax = plt.subplots(figsize=(10, 6))
for i, drift in enumerate(drifts):
    model = PhiMLP()
    opt = optim.Adam(model.parameters(), lr=0.01)
    loss_history = []
    
    for e in range(epochs):
        loss, mse = phi_loss(model, x, y, drift)
        opt.zero_grad()
        loss.backward()
        opt.step()
        # φ-sparsity prune every 10 epochs
        if (e + 1) % 10 == 0:
            with torch.no_grad():
                for p in model.parameters():
                    p.data[torch.abs(p.data) < 1/phi] = 0
        loss_history.append(mse)
    
    losses[drift] = loss_history
    ax.plot(range(epochs), loss_history, color=colors[i], label=f'Δη={drift}', linewidth=2)

ax.set_yscale('log')
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Toy AGI Training: Loss Poles from η-Drift')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axvline(x=50, color='r', linestyle='--', alpha=0.5, label='Pole Onset (>0.382 Drift)')
plt.savefig('agi_eta_drift.png', dpi=300, bbox_inches='tight')
plt.show()

# Print final losses for audit
for d in drifts:
    print(f'Δη={d}: Final Loss = {losses[d][-1]:.4f}')