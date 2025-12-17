import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
np.random.seed(42)

n = 2000
tickers = [f"ASSET{i:04d}" for i in range(n)]

# latent regimes
regime = np.random.choice(
    ["bull", "bear", "volatile"],
    size=n,
    p=[0.45, 0.35, 0.20]
)

momentum = []
volatility = []

for r in regime:
    if r == "bull":
        momentum.append(np.random.normal(0.04, 0.05))
        volatility.append(np.random.normal(0.04, 0.02))
    elif r == "bear":
        momentum.append(np.random.normal(-0.03, 0.06))
        volatility.append(np.random.normal(0.07, 0.03))
    else:
        momentum.append(np.random.normal(0.0, 0.12))
        volatility.append(np.random.normal(0.10, 0.04))

momentum = np.array(momentum)
volatility = np.abs(np.array(volatility))

pe_ratio = np.clip(np.random.normal(20, 10, n), 3, 250)
sector_signal = np.random.choice([0, 1, -1], size=n, p=[0.65, 0.2, 0.15])
liquidity = np.clip(np.random.beta(2, 5, n), 0.05, 1.0)

# decision score (same logic, richer signal)
score = (
    2.5 * momentum
    - 2.0 * volatility
    - 0.012 * (pe_ratio - 18)
    + 0.6 * sector_signal
    + 0.7 * liquidity
)

prob_buy = 1 / (1 + np.exp(-6 * (score - 0.015)))

action = np.where(
    prob_buy > 0.7, 1,
    np.where(prob_buy < 0.35, 2, 0)
)

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

print("Enhanced synthetic dataset generated.")
print(df.head())
