# (paste seed_data.py content provided below)
# backend/seed_data.py
import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)

n = 1500
tickers = [f"ASSET{i:03d}" for i in range(n)]

# features
momentum = np.random.normal(loc=0.02, scale=0.08, size=n)   # recent returns
volatility = np.abs(np.random.normal(loc=0.05, scale=0.03, size=n))
pe_ratio = np.clip(np.random.normal(loc=18, scale=8, size=n), 2, 200)
sector_signal = np.random.choice([0, 1, -1], size=n, p=[0.7, 0.15, 0.15])  # 1=positive sector, -1=negative
liquidity = np.random.uniform(0.1, 1.0, size=n)

# define a semi-realistic target: buy if momentum high, volatility low, good sector, reasonable PE
score = 2*momentum - 1.5*volatility - 0.01*(pe_ratio-15) + 0.5*sector_signal + 0.5*liquidity
prob_buy = 1 / (1 + np.exp(-5*(score - 0.02)))
rand = np.random.rand(n)
action = np.where(prob_buy > 0.7, 1, np.where(prob_buy < 0.35, 2, 0))  # 1 buy, 2 sell, 0 hold

df = pd.DataFrame({
    "ticker": tickers,
    "momentum": momentum,
    "volatility": volatility,
    "pe_ratio": pe_ratio,
    "sector_signal": sector_signal,
    "liquidity": liquidity,
    "action": action
})
df.to_csv("data/all_assets.csv", index=False)
print("Seeded data at data/all_assets.csv")
print(df.head())
