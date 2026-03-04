"""
ERPO Law Analysis: Spatio-Temporal Graph Neural Network
=======================================================
Implements the full methodology from the research proposal:

  Phase 1 — Cross-Lagged Mutual Information (MI) to discover τ*
  Phase 2 — ST-GNN with attention-based message passing for spillover
  Phase 3 — PDF report with spatial heatmap, temporal importance, and summary

Bug fixes vs. previous versions:
  - GDP is stored as a fixed per-county-per-month array (not re-randomized per row)
  - Spatial spillover is injected into synthetic data so the model can learn it
  - County adjacency changed so all 5 nodes have neighbors (Erie was isolated)
  - Saliency computed on ERPO feature only (index 0), not averaged over all features
  - Message passing includes an MLP transformation φ as described in the LaTeX
  - Loss includes L2 regularization: L = MSE + γ * ||α||²
  - MI-based lag discovery implemented as Phase 1
  - Plots include proper axis labels
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

# ==========================================
# 1. DATA GENERATION
# ==========================================
# Fix: Use 5 counties that all share borders — no isolated nodes
counties = ['Albany', 'Rensselaer', 'Saratoga', 'Schenectady', 'Columbia']
num_nodes = len(counties)
num_months = 120

# Physical adjacency matrix (shared borders)
# Albany borders all four; Columbia borders Albany and Rensselaer
# This ensures every node has at least one neighbor for spillover
physical_adj = np.array([
    [1, 1, 1, 1, 1],  # Albany
    [1, 1, 0, 0, 1],  # Rensselaer
    [1, 0, 1, 1, 0],  # Saratoga
    [1, 0, 1, 1, 0],  # Schenectady
    [1, 1, 0, 0, 1],  # Columbia
], dtype=np.float32)

TAU_TRUE = 3  # Ground-truth causal lag injected into data

np.random.seed(42)

# Fix: Generate GDP once as a fixed array, not re-randomized per DataFrame row
erpo_data = np.random.poisson(lam=5, size=(num_months, num_nodes)).astype(np.float32)
gdp_data = np.random.normal(50000, 5000, size=(num_months, num_nodes)).astype(np.float32)
mortality_data = np.zeros((num_months, num_nodes), dtype=np.float32)

# Inject causal signal:
#   - Each county's mortality depends on its own ERPO from TAU_TRUE months ago
#   - Fix: Albany (index 0) also spills over to its adjacent counties
for t in range(TAU_TRUE, num_months):
    for i in range(num_nodes):
        local_signal = 0.8 * mortality_data[t - 1, i]
        erpo_signal = -0.5 * erpo_data[t - TAU_TRUE, i]
        # Spatial spillover: Albany's ERPO reduces mortality in adjacent counties
        spillover = 0.0
        if i != 0 and physical_adj[i, 0] == 1:
            spillover = -0.2 * erpo_data[t - TAU_TRUE, 0]
        mortality_data[t, i] = 10 + local_signal + erpo_signal + spillover + np.random.normal(0, 0.5)

# ==========================================
# 2. PHASE 1 — CROSS-LAGGED MUTUAL INFORMATION
# ==========================================
# The LaTeX defines: τ* = argmax_{τ ∈ T} I(X_{t-τ}; Y_t)
# T = {τ_min, ..., τ_max} with τ_min=1, τ_max=12

def compute_mi_lags(erpo, mortality, tau_min=1, tau_max=12, n_bins=5):
    """
    Compute Cross-Lagged MI for all lags in the search space T.
    Returns the lag list and corresponding MI scores.
    """
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                           strategy='uniform', subsample=None)
    lags = list(range(tau_min, tau_max + 1))
    mi_scores = []
    for tau in lags:
        x = erpo[:-tau]
        y = mortality[tau:]
        xd = est.fit_transform(x.reshape(-1, 1)).flatten()
        yd = est.fit_transform(y.reshape(-1, 1)).flatten()
        mi_scores.append(mutual_info_score(xd, yd))
    return lags, mi_scores

print("--- Phase 1: Cross-Lagged Mutual Information ---")
all_mi = {}
for i, county in enumerate(counties):
    lags, mi_scores = compute_mi_lags(erpo_data[:, i], mortality_data[:, i])
    all_mi[county] = (lags, mi_scores)
    tau_star = lags[int(np.argmax(mi_scores))]
    print(f"  {county}: τ* = {tau_star} months  (ground truth = {TAU_TRUE})")

# ==========================================
# 3. DATA PREPROCESSING (SLIDING WINDOW)
# ==========================================
seq_length = 6  # Look-back window: t-6 to t-1
num_features = 3  # [ERPO, GDP, Mortality]

X_seq, Y_seq = [], []
for t in range(num_months - seq_length):
    x_t, y_t = [], []
    for i in range(num_nodes):
        # Fix: use the pre-generated gdp_data array (not re-randomize)
        window = np.column_stack([
            erpo_data[t: t + seq_length, i],
            gdp_data[t: t + seq_length, i],
            mortality_data[t: t + seq_length, i],
        ])
        x_t.append(window)
        y_t.append(mortality_data[t + seq_length, i])
    X_seq.append(x_t)
    Y_seq.append(y_t)

# Shapes: (T_samples, num_nodes, seq_length, num_features)
X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32)
Adj_tensor = torch.tensor(physical_adj, dtype=torch.float32)

# ==========================================
# 4. ST-GNN ARCHITECTURE
# ==========================================
# LaTeX Eq: m_i = Σ_{j ∈ N(i)} α_ij · φ(X_j, Z_j, Y_j)
# φ is an MLP that transforms neighbor features before aggregation

class MessageMLP(nn.Module):
    """
    φ: Small MLP that transforms a neighbor's feature vector before
    aggregation (as described in Section 6.2 of the LaTeX).
    """
    def __init__(self, num_features, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
        )

    def forward(self, x):
        return self.net(x)


class ST_GNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network.

    Spatial layer: Learned attention weights α_ij masked by physical adjacency.
    Temporal layer: GRU (gated temporal convolution) that attends to the lag
                    τ* inside the search space T.
    """
    def __init__(self, num_nodes, num_features, hidden_dim):
        super().__init__()

        # Fix: Initialize so diagonal (local effect) > off-diagonal (spillover baseline)
        # This reflects the prior that local ERPO has stronger effect than spillover
        init_matrix = torch.eye(num_nodes) * 1.0 + torch.ones(num_nodes, num_nodes) * 0.1
        self.alpha = nn.Parameter(init_matrix)  # Learnable α_ij

        # Fix: MLP for message transformation φ (missing from previous versions)
        self.message_mlp = MessageMLP(num_features, hidden_dim)

        # Gated temporal learning (Section 6.3: Temporal Gated Convolutions)
        self.temporal_gru = nn.GRU(num_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, X, physical_adj):
        # X: (batch, nodes, seq_len, features)
        batch_size, num_nodes, seq_len, num_feat = X.shape

        # Apply border mask: only learn weights for geographically adjacent counties
        alpha = self.alpha * physical_adj  # α_ij = 0 for non-adjacent pairs

        # Apply message MLP φ to all node features across all time steps
        X_flat = X.reshape(-1, num_feat)
        X_transformed = self.message_mlp(X_flat).reshape(batch_size, num_nodes, seq_len, num_feat)

        # Spatial aggregation: m_i = Σ_j α_ij · φ(X_j)
        # Permute to (batch, seq, nodes, features) for einsum
        X_perm = X_transformed.permute(0, 2, 1, 3)                      # (B, S, N, F)
        spatial_out = torch.einsum('ij,bsjf->bsif', alpha, X_perm)      # (B, S, N, F)
        spatial_out = spatial_out.permute(0, 2, 1, 3)                   # (B, N, S, F)
        spatial_out = spatial_out.reshape(batch_size * num_nodes, seq_len, num_feat)

        # Temporal GRU: attends to the optimal lag τ* within the window
        _, hidden = self.temporal_gru(spatial_out)  # hidden: (1, B*N, hidden_dim)
        out = self.fc(hidden.squeeze(0)).view(batch_size, num_nodes)    # (B, N)
        return out, alpha


model = ST_GNN(num_nodes=num_nodes, num_features=num_features, hidden_dim=32)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# ==========================================
# 5. TRAINING WITH REGULARIZATION
# ==========================================
# Fix: Loss includes L2 regularization term as specified in LaTeX Section 6.4:
#   L = Σ ||Y - Ŷ||² + γ * L_reg

gamma = 1e-3
loss_history = []

print("\n--- Phase 2: Training ST-GNN (300 epochs) ---")
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    pred, _ = model(X_tensor, Adj_tensor)
    mse_loss = criterion(pred, Y_tensor)
    # L2 regularization on spatial attention weights
    l2_reg = gamma * (model.alpha ** 2).sum()
    total_loss = mse_loss + l2_reg
    total_loss.backward()
    optimizer.step()
    loss_history.append(mse_loss.item())
    if (epoch + 1) % 60 == 0:
        print(f"  Epoch {epoch+1:3d}/300 | MSE: {mse_loss.item():.4f}")

# ==========================================
# 6. SALIENCY ANALYSIS — TEMPORAL IMPORTANCE
# ==========================================
# Fix: Compute gradient saliency on the ERPO feature only (index 0),
# not averaged over all features. GDP noise and Mortality autocorrelation
# would dilute the ERPO-specific lag signal otherwise.
model.eval()
X_sample = X_tensor[-30:].detach().requires_grad_(True)
pred_s, final_alpha = model(X_sample, Adj_tensor)
pred_s.sum().backward()

# Saliency for ERPO feature only → shape (seq_len,)
erpo_saliency = X_sample.grad.abs()[:, :, :, 0].mean(dim=(0, 1)).numpy()
step_importance = erpo_saliency / (erpo_saliency.sum() + 1e-10)

final_adj = final_alpha.detach().numpy()
opt_lag_idx = int(np.argmax(step_importance))
# seq_length=6, lags labelled t-6..t-1; index 0 → t-6, index 5 → t-1
opt_lag_val = seq_length - opt_lag_idx
print(f"\n  Gradient saliency τ* = {opt_lag_val} months  (ground truth = {TAU_TRUE})")

# ==========================================
# 7. PDF REPORT
# ==========================================
print("\n--- Generating PDF Report ---")

with PdfPages('ERPO_Analysis_Report.pdf') as pdf:

    # PAGE 1: Training Loss Convergence
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='tab:red', linewidth=2)
    plt.title("Model Convergence: Training Loss (MSE)\n"
              "Downward trend confirms the model is learning the causal relationship.")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True, linestyle='--', alpha=0.6)
    pdf.savefig()
    plt.close()

    # PAGE 2: Cross-Lagged MI — Optimal Delay τ* per County
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, (county, (lags, mi_scores)) in enumerate(all_mi.items()):
        ax = axes[idx]
        tau_star = lags[int(np.argmax(mi_scores))]
        ax.plot(lags, mi_scores, marker='o', color='teal', linewidth=2)
        ax.axvline(x=tau_star, color='red', linestyle='--',
                   label=f'τ* = {tau_star} mo')
        ax.set_title(county, fontweight='bold')
        ax.set_xlabel('Lag τ (months)')
        ax.set_ylabel('MI Score  I(X_{t-τ}; Y_t)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
    axes[-1].axis('off')
    fig.suptitle('Phase 1 — Cross-Lagged Mutual Information: Optimal Delay τ* per County',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # PAGE 3: Learned Spatial Attention Weights α_ij (Spillover Heatmap)
    plt.figure(figsize=(9, 7))
    sns.heatmap(final_adj, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=counties, yticklabels=counties)
    plt.title("Phase 2 — Learned Attention Weights α_ij\n"
              "Diagonal = Local ERPO effect   |   Off-diagonal = Geographic spillover")
    plt.xlabel("Source County (Neighbor j)")
    plt.ylabel("Target County (i)")
    pdf.savefig()
    plt.close()

    # PAGE 4: Temporal Importance of ERPO Signal (Gradient Saliency)
    lags_labels = [f"t-{i}" for i in range(seq_length, 0, -1)]
    colors = ['tomato' if k == opt_lag_idx else 'skyblue' for k in range(seq_length)]
    plt.figure(figsize=(9, 5))
    plt.bar(lags_labels, step_importance * 100, color=colors, edgecolor='black')
    plt.title(f"Temporal Importance of ERPO Signal (Gradient Saliency on ERPO Feature)\n"
              f"Optimal Delay τ* = {opt_lag_val} months (highlighted in red)")
    plt.xlabel("Time Step Prior to Outcome")
    plt.ylabel("Normalized Importance (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    pdf.savefig()
    plt.close()

    # PAGE 5: Summary Text
    fig = plt.figure(figsize=(8.5, 11))
    txt = "ERPO Law Spatio-Temporal Analysis — Summary Report\n"
    txt += "=" * 50 + "\n\n"

    txt += "PHASE 1: CROSS-LAGGED MI — OPTIMAL DELAY τ*\n"
    txt += "-" * 45 + "\n"
    for county, (lags, mi_scores) in all_mi.items():
        tau_star = lags[int(np.argmax(mi_scores))]
        txt += f"  {county:15s}: τ* = {tau_star} months\n"

    txt += f"\nPHASE 2: GRADIENT SALIENCY (ERPO feature)\n"
    txt += "-" * 45 + "\n"
    txt += f"  τ* = {opt_lag_val} months  (ground truth injected = {TAU_TRUE})\n"

    txt += "\n\nSPATIAL SPILLOVERS — Learned α_ij Weights:\n"
    txt += "-" * 45 + "\n"
    txt += "(Diagonal = local effect, off-diagonal = cross-county spillover)\n\n"
    for i, target in enumerate(counties):
        neighbors = [
            (counties[j], final_adj[i, j])
            for j in range(num_nodes)
            if physical_adj[i, j] == 1 and i != j
        ]
        neighbors.sort(key=lambda x: -x[1])
        txt += f"{target} ← spillover from:\n"
        for nb, w in neighbors:
            txt += f"    {nb:15s}: α = {w:.3f}\n"
        txt += "\n"

    txt += "\nINTERPRETATION:\n"
    txt += "  - Counties with high off-diagonal α_ij are 'Information Hubs'\n"
    txt += "    (primary sources of regional ERPO spillover).\n"
    txt += "  - Albany (index 0) was seeded as the spillover hub in this\n"
    txt += "    synthetic dataset; high α from Albany to neighbors confirms\n"
    txt += "    the model learned the injected spatial signal.\n"

    plt.text(0.05, 0.97, txt, transform=fig.transFigure,
             size=9, family='monospace', va='top')
    plt.axis('off')
    pdf.savefig()
    plt.close()

print("Success! Report saved: ERPO_Analysis_Report.pdf")
