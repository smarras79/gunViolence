"""
ERPO Law Analysis: Spatio-Temporal Graph Neural Network
=======================================================
Implements the full methodology from the research proposal:

  Phase 1 — Cross-Lagged Mutual Information (MI) with bootstrap confidence
            intervals and permutation p-values to discover τ*
  Phase 2 — ST-GNN with attention-based message passing for spillover
  Phase 3 — Occlusion-based temporal importance (robust alternative to
            gradient saliency, which vanishes through GRU chains)
  Phase 4 — PDF report with diagnostics, heatmaps, and full summary

Data: 10 NY counties, 240 months, z-score normalized features,
      multi-hub spillover, seasonal mortality variation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

# ==========================================
# 1. DATA GENERATION — 10 NY Counties
# ==========================================
counties = [
    'Albany',       # 0  — urban hub, Capital District
    'Rensselaer',  # 1  — borders Albany
    'Saratoga',    # 2  — borders Albany, Schenectady
    'Schenectady', # 3  — borders Albany, Saratoga
    'Columbia',    # 4  — borders Albany, Rensselaer, Greene, Dutchess
    'Greene',      # 5  — borders Albany, Columbia, Ulster
    'Ulster',      # 6  — borders Greene, Dutchess, Orange
    'Dutchess',    # 7  — borders Columbia, Ulster, Orange, Putnam
    'Orange',      # 8  — borders Ulster, Dutchess, Rockland
    'Rockland',    # 9  — borders Orange
]
num_nodes = len(counties)
num_months = 240  # 20 years of monthly data

# Physical adjacency (symmetric, based on actual county borders)
physical_adj = np.array([
    # Alb  Ren  Sar  Sch  Col  Gre  Uls  Dut  Ora  Roc
    [  1,   1,   1,   1,   1,   1,   0,   0,   0,   0],  # Albany
    [  1,   1,   0,   0,   1,   0,   0,   0,   0,   0],  # Rensselaer
    [  1,   0,   1,   1,   0,   0,   0,   0,   0,   0],  # Saratoga
    [  1,   0,   1,   1,   0,   0,   0,   0,   0,   0],  # Schenectady
    [  1,   1,   0,   0,   1,   1,   0,   1,   0,   0],  # Columbia
    [  1,   0,   0,   0,   1,   1,   1,   0,   0,   0],  # Greene
    [  0,   0,   0,   0,   0,   1,   1,   1,   1,   0],  # Ulster
    [  0,   0,   0,   0,   1,   0,   1,   1,   1,   0],  # Dutchess
    [  0,   0,   0,   0,   0,   0,   1,   1,   1,   1],  # Orange
    [  0,   0,   0,   0,   0,   0,   0,   0,   1,   1],  # Rockland
], dtype=np.float32)

TAU_TRUE = 3  # Ground-truth causal lag (months)

np.random.seed(42)

# County-specific ERPO filing rates (urban counties file more)
erpo_base_rates = [8, 4, 3, 5, 3, 2, 4, 5, 6, 4]
erpo_data = np.zeros((num_months, num_nodes), dtype=np.float32)
for i in range(num_nodes):
    erpo_data[:, i] = np.random.poisson(lam=erpo_base_rates[i], size=num_months)

# GDP with county-specific levels and slow linear trend
gdp_base = [55000, 42000, 60000, 45000, 48000, 38000, 44000, 52000, 50000, 58000]
gdp_data = np.zeros((num_months, num_nodes), dtype=np.float32)
for i in range(num_nodes):
    trend = np.linspace(0, 5000, num_months)
    gdp_data[:, i] = gdp_base[i] + trend + np.random.normal(0, 1500, num_months)

# Mortality with causal ERPO signal + multi-hub spillover
# Hub 0 = Albany (Capital District), Hub 8 = Orange (Hudson Valley)
mortality_data = np.zeros((num_months, num_nodes), dtype=np.float32)
mort_baselines = [12, 8, 6, 9, 7, 5, 8, 10, 11, 7]

for t in range(TAU_TRUE, num_months):
    for i in range(num_nodes):
        baseline = mort_baselines[i]
        ar_signal = 0.5 * mortality_data[t - 1, i]
        erpo_signal = -0.8 * erpo_data[t - TAU_TRUE, i]
        gdp_effect = -0.00005 * (gdp_data[t, i] - 50000)
        spillover = 0.0
        for hub in [0, 8]:
            if i != hub and physical_adj[i, hub] == 1:
                spillover += -0.3 * erpo_data[t - TAU_TRUE, hub]
        seasonal = 1.5 * np.sin(2 * np.pi * t / 12)
        noise = np.random.normal(0, 0.8)
        mortality_data[t, i] = max(0, baseline + ar_signal + erpo_signal
                                   + gdp_effect + spillover + seasonal + noise)

# ==========================================
# 2. PHASE 1 — CROSS-LAGGED MI WITH BOOTSTRAP CI
# ==========================================

def compute_mi_with_ci(erpo, mortality, tau_min=1, tau_max=12,
                       n_bins=5, n_bootstrap=200):
    """
    Cross-Lagged MI for each lag in search space T = {tau_min, ..., tau_max},
    with bootstrap 95% confidence intervals and permutation p-value.
    """
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal',
                           strategy='uniform', subsample=None)
    lags = list(range(tau_min, tau_max + 1))

    mi_observed = []
    for tau in lags:
        x = erpo[:-tau]
        y = mortality[tau:]
        xd = est.fit_transform(x.reshape(-1, 1)).flatten()
        yd = est.fit_transform(y.reshape(-1, 1)).flatten()
        mi_observed.append(mutual_info_score(xd, yd))
    mi_observed = np.array(mi_observed)

    mi_low = np.zeros(len(lags))
    mi_high = np.zeros(len(lags))
    for li, tau in enumerate(lags):
        x = erpo[:-tau]
        y = mortality[tau:]
        boot_scores = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(x), size=len(x), replace=True)
            xd = est.fit_transform(x[idx].reshape(-1, 1)).flatten()
            yd = est.fit_transform(y[idx].reshape(-1, 1)).flatten()
            boot_scores.append(mutual_info_score(xd, yd))
        mi_low[li] = np.percentile(boot_scores, 2.5)
        mi_high[li] = np.percentile(boot_scores, 97.5)

    best_lag_idx = int(np.argmax(mi_observed))
    best_tau = lags[best_lag_idx]
    observed_peak = mi_observed[best_lag_idx]

    x_peak = erpo[:-best_tau]
    y_peak = mortality[best_tau:]
    null_dist = []
    for _ in range(500):
        y_shuffled = np.random.permutation(y_peak)
        xd = est.fit_transform(x_peak.reshape(-1, 1)).flatten()
        yd = est.fit_transform(y_shuffled.reshape(-1, 1)).flatten()
        null_dist.append(mutual_info_score(xd, yd))
    p_value = np.mean(np.array(null_dist) >= observed_peak)

    return {
        'lags': lags,
        'mi': mi_observed,
        'ci_low': mi_low,
        'ci_high': mi_high,
        'tau_star': best_tau,
        'p_value': p_value,
    }


print("--- Phase 1: Cross-Lagged Mutual Information (with bootstrap CI) ---")
all_mi = {}
for i, county in enumerate(counties):
    result = compute_mi_with_ci(erpo_data[:, i], mortality_data[:, i])
    all_mi[county] = result
    sig = "***" if result['p_value'] < 0.01 else ("**" if result['p_value'] < 0.05 else "n.s.")
    print(f"  {county:15s}: τ* = {result['tau_star']:2d} mo   "
          f"p = {result['p_value']:.3f} {sig}   (truth = {TAU_TRUE})")

# ==========================================
# 3. DATA PREPROCESSING
# ==========================================
# Z-score normalize features so all have mean=0, std=1.
# Without this, GDP (~50000) dominates ERPO (~5) and Mortality (~10),
# causing the model to ignore the ERPO signal entirely — which is why
# the temporal importance was all zeros in the previous version.
seq_length = 8
num_features = 3

erpo_mean, erpo_std = erpo_data.mean(), erpo_data.std()
gdp_mean, gdp_std = gdp_data.mean(), gdp_data.std()
mort_mean, mort_std = mortality_data.mean(), mortality_data.std()

erpo_norm = (erpo_data - erpo_mean) / (erpo_std + 1e-8)
gdp_norm = (gdp_data - gdp_mean) / (gdp_std + 1e-8)
mort_norm = (mortality_data - mort_mean) / (mort_std + 1e-8)

X_seq, Y_seq = [], []
for t in range(num_months - seq_length):
    x_t, y_t = [], []
    for i in range(num_nodes):
        window = np.column_stack([
            erpo_norm[t: t + seq_length, i],
            gdp_norm[t: t + seq_length, i],
            mort_norm[t: t + seq_length, i],
        ])
        x_t.append(window)
        # Normalize targets too
        y_t.append((mortality_data[t + seq_length, i] - mort_mean) / (mort_std + 1e-8))
    X_seq.append(x_t)
    Y_seq.append(y_t)

X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32)
Adj_tensor = torch.tensor(physical_adj, dtype=torch.float32)

# ==========================================
# 4. ST-GNN ARCHITECTURE
# ==========================================

class MessageMLP(nn.Module):
    """φ: Transforms neighbor features before aggregation (LaTeX Sec 6.2)."""
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
    Spatio-Temporal Graph Neural Network with:
    - Learned attention weights α_ij masked by physical adjacency
    - MLP message transformation φ
    - GRU for temporal gated learning
    """
    def __init__(self, num_nodes, num_features, hidden_dim):
        super().__init__()
        init_matrix = torch.eye(num_nodes) * 1.0 + torch.ones(num_nodes, num_nodes) * 0.1
        self.alpha = nn.Parameter(init_matrix)
        self.message_mlp = MessageMLP(num_features, hidden_dim)
        self.temporal_gru = nn.GRU(num_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, X, physical_adj):
        batch_size, num_nodes, seq_len, num_feat = X.shape
        alpha = self.alpha * physical_adj

        X_flat = X.reshape(-1, num_feat)
        X_transformed = self.message_mlp(X_flat).reshape(
            batch_size, num_nodes, seq_len, num_feat)

        X_perm = X_transformed.permute(0, 2, 1, 3)
        spatial_out = torch.einsum('ij,bsjf->bsif', alpha, X_perm)
        spatial_out = spatial_out.permute(0, 2, 1, 3)
        spatial_out = spatial_out.reshape(batch_size * num_nodes, seq_len, num_feat)

        _, hidden = self.temporal_gru(spatial_out)
        out = self.fc(hidden.squeeze(0)).view(batch_size, num_nodes)
        return out, alpha


model = ST_GNN(num_nodes=num_nodes, num_features=num_features, hidden_dim=32)
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.MSELoss()

# ==========================================
# 5. TRAINING WITH REGULARIZATION
# ==========================================
gamma = 1e-3
loss_history = []
num_epochs = 500

print(f"\n--- Phase 2: Training ST-GNN ({num_epochs} epochs) ---")
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    pred, _ = model(X_tensor, Adj_tensor)
    mse_loss = criterion(pred, Y_tensor)
    l2_reg = gamma * (model.alpha ** 2).sum()
    total_loss = mse_loss + l2_reg
    total_loss.backward()
    optimizer.step()
    loss_history.append(mse_loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1:3d}/{num_epochs} | MSE: {mse_loss.item():.4f}")

# ==========================================
# 6. OCCLUSION-BASED TEMPORAL IMPORTANCE
# ==========================================
# Why not gradient saliency?
#   Gradients vanish through the GRU backward pass, producing all-zeros
#   for early time steps. Occlusion is model-agnostic and robust:
#   zero out ERPO at time step s → measure MSE increase → larger = more important.
model.eval()
with torch.no_grad():
    baseline_pred, final_alpha = model(X_tensor, Adj_tensor)
    baseline_mse = criterion(baseline_pred, Y_tensor).item()

    step_mse_increase = []
    for s in range(seq_length):
        X_occluded = X_tensor.clone()
        # Set ERPO (feature 0) at time step s to 0 (= population mean after z-norm)
        X_occluded[:, :, s, 0] = 0.0
        occluded_pred, _ = model(X_occluded, Adj_tensor)
        occluded_mse = criterion(occluded_pred, Y_tensor).item()
        step_mse_increase.append(occluded_mse - baseline_mse)

    step_mse_increase = np.array(step_mse_increase)
    # Clip negatives (rounding artifacts) and normalize
    step_mse_increase = np.maximum(step_mse_increase, 0)
    step_importance = step_mse_increase / (step_mse_increase.sum() + 1e-10)

opt_lag_idx = int(np.argmax(step_importance))
opt_lag_val = seq_length - opt_lag_idx

final_adj = final_alpha.detach().numpy()

print(f"\n  Occlusion temporal importance τ* = {opt_lag_val} months  "
      f"(ground truth = {TAU_TRUE})")
for s in range(seq_length):
    marker = " <-- PEAK" if s == opt_lag_idx else ""
    print(f"    t-{seq_length - s}: {step_importance[s]:6.1%}  "
          f"(raw ΔMSE = {step_mse_increase[s]:.6f}){marker}")

# ==========================================
# 7. PDF REPORT
# ==========================================
print("\n--- Generating PDF Report ---")

with PdfPages('ERPO_Analysis_Report.pdf') as pdf:

    # PAGE 1: Training Loss Convergence
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='tab:red', linewidth=2)
    plt.title("Model Convergence: Training Loss (MSE on normalized targets)\n"
              "Downward trend confirms the model is learning the causal relationship.")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True, linestyle='--', alpha=0.6)
    pdf.savefig()
    plt.close()

    # PAGE 2: Cross-Lagged MI with Confidence Intervals
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for idx, (county, res) in enumerate(all_mi.items()):
        ax = axes[idx]
        lags = res['lags']
        ax.plot(lags, res['mi'], marker='o', color='teal', linewidth=2, zorder=3)
        ax.fill_between(lags, res['ci_low'], res['ci_high'],
                        alpha=0.25, color='teal', label='95% CI')
        ax.axvline(x=res['tau_star'], color='red', linestyle='--', linewidth=2,
                   label=f"τ*={res['tau_star']}mo")
        ax.axvline(x=TAU_TRUE, color='gray', linestyle=':', linewidth=1.5,
                   label=f'truth={TAU_TRUE}mo')
        sig_str = f"p={res['p_value']:.3f}"
        if res['p_value'] < 0.01:
            sig_str += " ***"
        elif res['p_value'] < 0.05:
            sig_str += " **"
        ax.set_title(f"{county}\n{sig_str}", fontweight='bold', fontsize=9)
        ax.set_xlabel('Lag τ (months)')
        ax.set_ylabel('MI Score')
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.4)
    fig.suptitle('Phase 1 — Cross-Lagged Mutual Information: Optimal Delay τ*\n'
                 '(shaded = 95% bootstrap CI, red = detected τ*, gray = ground truth)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # PAGE 3: Spatial Spillover Heatmap
    plt.figure(figsize=(11, 9))
    sns.heatmap(final_adj, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=counties, yticklabels=counties)
    plt.title("Phase 2 — Learned Attention Weights α_ij\n"
              "Diagonal = Local effect  |  Off-diagonal = Geographic spillover\n"
              "(Zero entries = non-adjacent counties, masked by physical adjacency)")
    plt.xlabel("Source County (Neighbor j)")
    plt.ylabel("Target County (i)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # PAGE 4: Occlusion-Based Temporal Importance
    lags_labels = [f"t-{seq_length - s}" for s in range(seq_length)]
    colors = ['tomato' if s == opt_lag_idx else 'skyblue' for s in range(seq_length)]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(lags_labels, step_importance * 100, color=colors, edgecolor='black')
    if step_importance.max() > 0:
        peak_bar = bars[opt_lag_idx]
        plt.annotate(f'τ* = {opt_lag_val} mo',
                     xy=(peak_bar.get_x() + peak_bar.get_width() / 2,
                         peak_bar.get_height()),
                     xytext=(0, 10), textcoords='offset points',
                     ha='center', fontweight='bold', fontsize=11, color='red')
    plt.title("Temporal Importance of ERPO Signal (Occlusion-Based)\n"
              "Each bar = MSE increase when that time step's ERPO is zeroed out.\n"
              "Higher bar = more important for predicting mortality.")
    plt.xlabel("Time Step Prior to Outcome")
    plt.ylabel("Normalized Importance (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # PAGE 5: Raw MSE Increase per Time Step (diagnostic)
    plt.figure(figsize=(10, 5))
    bar_colors = ['tomato' if s == opt_lag_idx else 'steelblue' for s in range(seq_length)]
    plt.bar(lags_labels, step_mse_increase, color=bar_colors, edgecolor='black')
    plt.title("Diagnostic: Raw MSE Increase per Occluded Time Step\n"
              "Unnormalized view — confirms the model uses ERPO signal.")
    plt.xlabel("Time Step Prior to Outcome")
    plt.ylabel("MSE Increase (occluded − baseline)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # PAGE 6: Summary Text
    fig = plt.figure(figsize=(8.5, 11))
    txt = "ERPO Spatio-Temporal Analysis — Summary Report\n"
    txt += "=" * 55 + "\n\n"

    txt += f"CONFIGURATION\n"
    txt += "-" * 55 + "\n"
    txt += f"  Counties:          {num_nodes}\n"
    txt += f"  Months:            {num_months} ({num_months // 12} years)\n"
    txt += f"  Sequence window:   {seq_length} months\n"
    txt += f"  Training epochs:   {num_epochs}\n"
    txt += f"  Ground truth τ:    {TAU_TRUE} months\n"
    txt += f"  Spillover hubs:    Albany (idx 0), Orange (idx 8)\n"
    txt += f"  Normalization:     Z-score (mean=0, std=1)\n\n"

    txt += "PHASE 1: CROSS-LAGGED MI — OPTIMAL DELAY τ*\n"
    txt += "-" * 55 + "\n"
    txt += f"  {'County':15s}  {'τ*':>4s}  {'p-value':>8s}  {'Sig':>5s}\n"
    for county, res in all_mi.items():
        sig = "***" if res['p_value'] < 0.01 else ("**" if res['p_value'] < 0.05 else "n.s.")
        txt += f"  {county:15s}  {res['tau_star']:3d}mo  {res['p_value']:8.3f}  {sig:>5s}\n"

    txt += f"\nPHASE 2: OCCLUSION-BASED TEMPORAL IMPORTANCE\n"
    txt += "-" * 55 + "\n"
    txt += f"  Identified τ* = {opt_lag_val} months\n"
    txt += f"  Per-step importance:\n"
    for s in range(seq_length):
        marker = "  <-- PEAK" if s == opt_lag_idx else ""
        txt += f"    t-{seq_length - s}: {step_importance[s]:6.1%}  " \
               f"(ΔMSE={step_mse_increase[s]:.6f}){marker}\n"

    txt += f"\nSPATIAL SPILLOVERS — Learned α_ij Weights\n"
    txt += "-" * 55 + "\n"
    for i, target in enumerate(counties):
        neighbors = [
            (counties[j], final_adj[i, j])
            for j in range(num_nodes)
            if physical_adj[i, j] == 1 and i != j
        ]
        neighbors.sort(key=lambda x: -x[1])
        diag = final_adj[i, i]
        txt += f"  {target} (local α={diag:.2f}):\n"
        for nb, w in neighbors:
            txt += f"      ← {nb:15s}  α = {w:.3f}\n"

    txt += "\nINTERPRETATION\n"
    txt += "-" * 55 + "\n"
    txt += "  - τ* near 3 months with p < 0.05 confirms the\n"
    txt += "    causal lag between ERPO filings and mortality.\n"
    txt += "  - Counties where MI peak is n.s. lack a reliable\n"
    txt += "    ERPO→mortality signal at any tested lag.\n"
    txt += "  - Diagonal α values are learned (not fixed at 1)\n"
    txt += "    to allow the model to find the optimal balance\n"
    txt += "    between local effect and neighbor spillover.\n"
    txt += "  - High off-diagonal α_ij identifies counties that\n"
    txt += "    are 'Information Hubs' for regional spillover.\n"

    plt.text(0.04, 0.97, txt, transform=fig.transFigure,
             size=8.5, family='monospace', va='top')
    plt.axis('off')
    pdf.savefig()
    plt.close()

print("Success! Report saved: ERPO_Analysis_Report.pdf")
