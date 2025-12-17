# app.py
import streamlit as st
import pandas as pd
import json

from backend.orchestrator import NeuroQuantMCP
from backend.utils import get_random_asset_sample

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(
    page_title="FinSight MCP Demo",
    layout="wide"
)

st.title("FinSight — MCP (Model • Context • Protocol) Demo")
st.markdown(
    "Offline, interview-ready FinTech demo integrating **NeuroQuant research logic** "
    "with MCP (Model–Context–Protocol) reasoning."
)

# --------------------------------------------------
# Instantiate MCP orchestrator
# --------------------------------------------------
mcp = NeuroQuantMCP()

# --------------------------------------------------
# Sidebar: Context
# --------------------------------------------------
st.sidebar.header("User Context")

risk = st.sidebar.selectbox("Risk tolerance", ["low", "medium", "high"], index=1)
exposure = st.sidebar.slider("Portfolio exposure", 0.0, 1.0, 0.25)
sentiment = st.sidebar.slider("Market sentiment", -1.0, 1.0, 0.0)
horizon = st.sidebar.selectbox("Investment horizon", ["short", "medium", "long"], index=1)

context = {
    "risk_tolerance": risk,
    "position_exposure": exposure,
    "market_sentiment": sentiment,
    "time_horizon": horizon
}

# --------------------------------------------------
# Asset selection
# --------------------------------------------------
st.sidebar.header("Asset Selection")

sample = get_random_asset_sample()

if st.sidebar.button("Pick random asset"):
    sample = get_random_asset_sample()

ticker = st.sidebar.text_input("Ticker", sample["ticker"])

momentum = st.sidebar.number_input(
    "Momentum",
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
    "Liquidity",
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
# Display
# --------------------------------------------------
st.subheader(f"Asset: {ticker}")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Asset Features")
    st.table(pd.DataFrame([features]).T.rename(columns={0: "value"}))

with col2:
    st.markdown("### Context Snapshot")
    st.json(context)

# --------------------------------------------------
# Run MCP
# --------------------------------------------------
if st.button("Run MCP"):
    result = mcp.run(features, context)
    decision = result["decision"]

    st.markdown("## Recommendation")

    if decision["action"] == "buy":
        st.success(f"BUY — confidence {decision['confidence']}")
    elif decision["action"] == "sell":
        st.error(f"SELL — confidence {decision['confidence']}")
    else:
        st.info(f"HOLD — confidence {decision['confidence']}")

    st.markdown("### Explanation")
    for r in decision["reasons"]:
        st.write(f"- {r}")

    st.markdown("### Adjusted Probabilities")
    st.dataframe(pd.DataFrame(result["adjusted_probs"], index=["probability"]).T)

    st.markdown("### Raw Model Output")
    st.dataframe(pd.DataFrame(result["raw_model_probs"], index=["probability"]).T)

    export = {
        "ticker": ticker,
        "features": features,
        "context": context,
        "decision": decision
    }

    st.download_button(
        "Download Recommendation JSON",
        json.dumps(export, indent=2),
        file_name=f"{ticker}_recommendation.json",
        mime="application/json"
    )
