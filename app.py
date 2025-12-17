# app.py
import streamlit as st
import pandas as pd
import json
import numpy as np

from backend.orchestrator import NeuroQuantMCP
from backend.utils import get_random_asset_sample

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(
    page_title="FinSight – NeuroQuant MCP",
    layout="wide",
    page_icon="📈"
)

st.title("📈 FinSight — NeuroQuant MCP Decision Engine")
st.markdown(
    """
    **Research-backed FinTech decision system** integrating  
    **NeuroQuant (Regime Detection + Diffusion Policy)**  
    with **MCP (Model–Context–Protocol)** reasoning.
    """
)

st.divider()

# --------------------------------------------------
# Instantiate MCP orchestrator
# --------------------------------------------------
mcp = NeuroQuantMCP()

# --------------------------------------------------
# Sidebar — User Context
# --------------------------------------------------
st.sidebar.header("🧠 Investor Context")

risk = st.sidebar.selectbox("Risk tolerance", ["low", "medium", "high"], index=1)
exposure = st.sidebar.slider("Current portfolio exposure", 0.0, 1.0, 0.25)
sentiment = st.sidebar.slider("Market sentiment", -1.0, 1.0, 0.0)
horizon = st.sidebar.selectbox("Investment horizon", ["short", "medium", "long"], index=1)

context = {
    "risk_tolerance": risk,
    "position_exposure": exposure,
    "market_sentiment": sentiment,
    "time_horizon": horizon
}

# --------------------------------------------------
# Sidebar — Asset Selection
# --------------------------------------------------
st.sidebar.header("📊 Asset Selection")

sample = get_random_asset_sample()

if st.sidebar.button("🎲 Pick random asset"):
    sample = get_random_asset_sample()

ticker = st.sidebar.text_input("Ticker", sample["ticker"])

momentum = st.sidebar.number_input(
    "Momentum (recent return)",
    value=float(sample["features"]["momentum"]),
    format="%.4f"
)
volatility = st.sidebar.number_input(
    "Volatility",
    value=float(sample["features"]["volatility"]),
    format="%.4f"
)
pe_ratio = st.sidebar.number_input(
    "P/E Ratio",
    value=float(sample["features"]["pe_ratio"]),
    step=1.0
)
sector_signal = st.sidebar.selectbox(
    "Sector signal",
    [-1, 0, 1],
    index=[-1, 0, 1].index(sample["features"]["sector_signal"])
)
liquidity = st.sidebar.number_input(
    "Liquidity (0–1)",
    min_value=0.0,
    max_value=1.0,
    value=float(sample["features"]["liquidity"]),
    format="%.4f"
)

features = {
    "momentum": momentum,
    "volatility": volatility,
    "pe_ratio": pe_ratio,
    "sector_signal": sector_signal,
    "liquidity": liquidity
}

# --------------------------------------------------
# Asset Overview
# --------------------------------------------------
st.subheader(f"🧾 Asset: `{ticker}`")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Momentum", f"{momentum:.2%}")
k2.metric("Volatility", f"{volatility:.2%}")
k3.metric("Liquidity", f"{liquidity:.2f}")
k4.metric("Sector Signal", sector_signal)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🔍 Asset Features")
    st.table(pd.DataFrame([features]).T.rename(columns={0: "value"}))

with col2:
    st.markdown("### 🧠 Context Snapshot")
    st.json(context)

st.divider()

# --------------------------------------------------
# Run NeuroQuant MCP
# --------------------------------------------------
if st.button("🚀 Run NeuroQuant MCP Analysis"):
    # Simulated return window for regime detection (demo-safe)
    return_window = np.random.normal(
        loc=features["momentum"],
        scale=max(features["volatility"], 1e-4),
        size=30
    )

    result = mcp.run(features, context, return_window)
    decision = result["decision"]

    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------
    st.markdown("## ✅ Recommendation")

    action = decision["action"]
    confidence = decision["confidence"]

    if action == "buy":
        st.success(f"**BUY** — confidence `{confidence}`")
    elif action == "sell":
        st.error(f"**SELL** — confidence `{confidence}`")
    else:
        st.info(f"**HOLD** — confidence `{confidence}`")

    # --------------------------------------------------
    # Explanation
    # --------------------------------------------------
    st.markdown("### 🧠 Explanation")
    for r in result["explanation"]:
        st.write(f"- {r}")

    # --------------------------------------------------
    # Regime Info
    # --------------------------------------------------
    st.markdown("### 🌐 Market Regime Detected")
    st.info(result["regime"])

    # --------------------------------------------------
    # Probability Visuals
    # --------------------------------------------------
    st.markdown("### 📊 Smoothed Decision Probabilities")
    prob_df = pd.DataFrame(result["smoothed_probs"], index=["Probability"]).T
    st.bar_chart(prob_df)

    st.markdown("### 🤖 Raw Model Output")
    raw_df = pd.DataFrame(result["raw_probs"], index=["Probability"]).T
    st.dataframe(raw_df)

    # --------------------------------------------------
    # Decision Summary
    # --------------------------------------------------
    st.markdown("### 🧾 Decision Summary")
    st.info(
        f"""
        **Final Action:** {action.upper()}  
        **Confidence:** {confidence:.2f}

        This decision is derived from:
        - Machine learning prediction (RandomForest)
        - Wasserstein-based regime detection (NeuroQuant)
        - Diffusion-smoothed policy update
        - Risk-aware MCP protocol
        """
    )

    # --------------------------------------------------
    # Export
    # --------------------------------------------------
    export = {
        "ticker": ticker,
        "features": features,
        "context": context,
        "regime": result["regime"],
        "decision": decision
    }

    st.download_button(
        "⬇️ Download Recommendation (JSON)",
        json.dumps(export, indent=2),
        file_name=f"{ticker}_recommendation.json",
        mime="application/json"
    )

else:
    st.info("Click **Run NeuroQuant MCP Analysis** to generate a regime-aware recommendation.")
