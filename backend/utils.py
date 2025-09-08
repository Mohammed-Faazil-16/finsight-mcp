# (paste utils.py content provided below)
# backend/utils.py
import pandas as pd
import numpy as np

def get_random_asset_sample(path="data/all_assets.csv"):
    df = pd.read_csv(path)
    row = df.sample(1).iloc[0]
    features = {
        "momentum": float(row["momentum"]),
        "volatility": float(row["volatility"]),
        "pe_ratio": float(row["pe_ratio"]),
        "sector_signal": int(row["sector_signal"]),
        "liquidity": float(row["liquidity"])
    }
    return {"ticker": row["ticker"], "features": features}
