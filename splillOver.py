import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

# 1. SETUP DATA
np.random.seed(42)
counties = ['Albany', 'Erie', 'Westchester', 'Monroe', 'Nassau']
months = pd.date_range(start='2019-01-01', periods=60, freq='ME')

data_list = []
for c in counties:
    for m in months:
        data_list.append({
            'county': c,
            'date': m,
            'erpo_treatment': np.random.poisson(lam=10),
            'mortality': np.random.poisson(lam=20),
            'gdp_proxy': np.random.normal(55000, 8000)
        })

df = pd.DataFrame(data_list)

# 2. VISUALIZING THE OPTIMAL DELAY (Mutual Information)
def plot_optimal_delay(data, target_county, max_lag=12):
    county_df = data[data['county'] == target_county].copy()
    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform', subsample=None)
    
    lags = []
    mi_scores = []
    
    for lag in range(1, max_lag + 1):
        shifted_mortality = county_df['mortality'].shift(-lag).dropna()
        treatment = county_df['erpo_treatment'].iloc[:-lag]
        
        x = est.fit_transform(treatment.values.reshape(-1, 1)).flatten()
        y = est.fit_transform(shifted_mortality.values.reshape(-1, 1)).flatten()
        
        mi = mutual_info_score(x, y)
        lags.append(lag)
        mi_scores.append(mi)
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=lags, y=mi_scores, marker='o', color='teal')
    plt.title(f"Finding the Optimal Lag for {target_county}")
    plt.xlabel("Delay in Months")
    plt.ylabel("Mutual Information Score (Strength of Relationship)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# 3. VISUALIZING SPILLOVERS AND COVARIATES
def plot_spillover_importance(data, target_county, lag=4):
    # Prepare spillover columns
    pivoted = data.pivot(index='date', columns='county', values='erpo_treatment')
    pivoted.columns = [f'ERPO_{c}' for c in pivoted.columns]
    merged = data[data['county'] == target_county].merge(pivoted, on='date')
    
    merged['target_mortality'] = merged['mortality'].shift(-lag)
    train_df = merged.dropna()
    
    # Select features (Local ERPOs, Neighbor ERPOs, and GDP)
    features = [col for col in train_df.columns if 'ERPO' in col or 'gdp' in col]
    X = train_df[features]
    y = train_df['target_mortality']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create the Importance Plot
    importance_df = pd.DataFrame({
        'Factor': features,
        'Influence': model.feature_importances_
    }).sort_values(by='Influence', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Influence', y='Factor', data=importance_df, palette='viridis')
    plt.title(f"What influences Mortality in {target_county}? (Lag={lag}mo)")
    plt.tight_layout()
    plt.show()

# RUN THE VISUALS
plot_optimal_delay(df, 'Albany')
plot_spillover_importance(df, 'Albany')
