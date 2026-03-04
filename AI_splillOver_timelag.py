import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# ==========================================
# 1. DATA GENERATION (With Stronger Causal Signal)
# ==========================================
counties = ['Albany', 'Rensselaer', 'Saratoga', 'Schenectady', 'Erie']
num_nodes = len(counties)
num_months = 120 

physical_adj = np.array([
    [1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 1]
], dtype=np.float32)

np.random.seed(42)
# Generate features
erpo_data = np.random.poisson(lam=5, size=(num_months, num_nodes)).astype(np.float32)
mortality_data = np.zeros((num_months, num_nodes), dtype=np.float32)

# Strong causal signal: Mortality is 80% own history + 20% ERPO lag
for t in range(3, num_months):
    for i in range(num_nodes):
        # Local history effect (The diagonal signal)
        local_signal = 0.8 * mortality_data[t-1, i]
        # ERPO effect from 3 months ago (The tau signal)
        policy_signal = -0.5 * erpo_data[t-3, i]
        mortality_data[t, i] = 10 + local_signal + policy_signal + np.random.normal(0, 0.5)

# Prep DataFrame
data_list = []
for t in range(num_months):
    for i in range(num_nodes):
        data_list.append({
            'month': t, 'county_idx': i,
            'ERPO': erpo_data[t, i], 'Mortality': mortality_data[t, i], 'GDP': np.random.normal(50000, 5000)
        })
df = pd.DataFrame(data_list)

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
seq_length = 6
X_seq, Y_seq = [], []
for t in range(num_months - seq_length):
    x_t, y_t = [], []
    for i in range(num_nodes):
        c_data = df[df['county_idx'] == i]
        x_t.append(c_data.iloc[t : t+seq_length][['ERPO', 'GDP', 'Mortality']].values)
        y_t.append(c_data.iloc[t+seq_length]['Mortality'])
    X_seq.append(x_t); Y_seq.append(y_t)

X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32)
Adj_tensor = torch.tensor(physical_adj, dtype=torch.float32)

# ==========================================
# 3. CORRECTED ST-GNN ARCHITECTURE
# ==========================================
class ST_GNN(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_dim):
        super(ST_GNN, self).__init__()
        # Initialize diagonal to 1.0 (Local) and others to 0.1 (Spillover)
        init_matrix = torch.eye(num_nodes) + (torch.ones(num_nodes, num_nodes) * 0.1)
        self.spatial_weights = nn.Parameter(init_matrix)
        self.temporal_gru = nn.GRU(num_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, X, physical_adj):
        batch_size, num_nodes, seq_len, _ = X.shape
        # Apply border mask
        learned_adj = self.spatial_weights * physical_adj
        
        X_reshaped = X.permute(0, 2, 1, 3) 
        spatial_out = torch.einsum('ij, bsjf -> bsif', learned_adj, X_reshaped).permute(0, 2, 1, 3)
        spatial_out = spatial_out.reshape(batch_size * num_nodes, seq_len, -1)
        
        _, hidden = self.temporal_gru(spatial_out)
        return self.fc(hidden.squeeze(0)).view(batch_size, num_nodes), learned_adj

model = ST_GNN(num_nodes, num_features=3, hidden_dim=16)
optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# Training loop
for epoch in range(300):
    model.train(); optimizer.zero_grad()
    pred, _ = model(X_tensor, Adj_tensor)
    loss = criterion(pred, Y_tensor)
    loss.backward(); optimizer.step()

# ==========================================
# 4. ANALYSIS & PDF EXPORT
# ==========================================
model.eval()
# Calculate Saliency
X_sample = X_tensor[-20:].detach().requires_grad_(True)
pred, final_adj_tensor = model(X_sample, Adj_tensor)
pred.sum().backward()
saliency = X_sample.grad.abs().mean(dim=(0, 1, 3)).numpy()
step_importance = saliency / (saliency.sum() + 1e-10)

final_adj = final_adj_tensor.detach().numpy()

with PdfPages('ERPO_Analysis_Report.pdf') as pdf:
    # PAGE 1: Spatial Heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(final_adj, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=counties, yticklabels=counties)
    plt.title("Learned Spatial Influence\n(Diagonal = Local Effect, Off-Diagonal = Spillover)")
    pdf.savefig(); plt.close()

    # PAGE 2: Temporal Importance
    plt.figure(figsize=(8, 5))
    lags = [f"t-{i}" for i in range(seq_length, 0, -1)]
    plt.bar(lags, step_importance, color='skyblue', edgecolor='black')
    plt.title("Temporal Search Space: Optimal Delay (tau*)")
    pdf.savefig(); plt.close()

    # PAGE 3: Summary Text
    summary_page = plt.figure(figsize=(8.5, 11)); summary_page.clf()
    opt_lag_val = seq_length - np.argmax(step_importance)
    txt = "ERPO Law Spatio-Temporal Analysis Report\n" + "="*45 + "\n\n"
    txt += f"IDENTIFIED OPTIMAL DELAY (tau*): {opt_lag_val} Months\n\n"
    txt += "INTERPRETATION:\n"
    txt += "1. Diagonal weights > 0.5 indicate strong local policy impact.\n"
    txt += "2. Off-diagonal weights indicate geographic spillovers.\n"
    plt.text(0.1, 0.95, txt, transform=summary_page.transFigure, size=10, family='monospace', va='top')
    pdf.savefig(); plt.close()

print("Success! Report saved as: ERPO_Analysis_Report.pdf")
