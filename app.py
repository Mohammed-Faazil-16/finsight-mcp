# (paste Streamlit app.py content provided below)
# app.py
import streamlit as st
from backend.mcp import ModelContextProtocol
from backend.utils import get_random_asset_sample
import pandas as pd
import json

st.set_page_config(page_title="FinSight MCP Demo", layout="wide")
st.title("FinSight — MCP (Model • Context • Protocol) Demo")
st.markdown("Offline, interview-ready FinTech demo: model + context + protocol producing actionable recommendations.")

# instantiate orchestrator
mcp = ModelContextProtocol()

# Sidebar: user context controls
st.sidebar.header("User Context (Context)")
risk = st.sidebar.selectbox("Risk tolerance", ["low", "medium", "high"], index=1)
exposure = st.sidebar.slider("Current portfolio exposure (0 = low, 1 = high)", 0.0, 1.0, 0.25)
sentiment = st.sidebar.slider("Market sentiment (-1 negative → +1 positive)", -1.0, 1.0, 0.0)
horizon = st.sidebar.selectbox("Investment horizon", ["short", "medium", "long"], index=1)

context = {
    "risk_tolerance": risk,
    "position_exposure": exposure,
    "market_sentiment": sentiment,
    "time_horizon": horizon
}

# Choose an asset to analyze
st.sidebar.header("Asset Selection")
sample = get_random_asset_sample()
if st.sidebar.button("Pick a random asset"):
    sample = get_random_asset_sample()

ticker = st.sidebar.text_input("Ticker", value=sample["ticker"])
# allow user to override features manually
st.sidebar.markdown("Override asset features (optional)")
momentum = st.sidebar.number_input("Momentum (recent return)", value=float(sample["features"]["momentum"]), format="%.4f")
volatility = st.sidebar.number_input("Volatility", value=float(sample["features"]["volatility"]), format="%.4f")
pe_ratio = st.sidebar.number_input("P/E Ratio", value=float(sample["features"]["pe_ratio"]), step=1.0)
sector_signal = st.sidebar.selectbox("Sector signal", [-1, 0, 1], index=[-1,0,1].index(sample["features"]["sector_signal"]))
liquidity = st.sidebar.number_input("Liquidity (0..1)", min_value=0.0, max_value=1.0, value=float(sample["features"]["liquidity"]), format="%.4f")

features = {
    "momentum": momentum,
    "volatility": volatility,
    "pe_ratio": pe_ratio,
    "sector_signal": sector_signal,
    "liquidity": liquidity
}

st.subheader(f"Asset: {ticker}")
col1, col2 = st.columns([2,1])

with col1:
    st.markdown("### Features")
    df_feat = pd.DataFrame([features]).T.rename(columns={0:"value"})
    st.table(df_feat)

with col2:
    st.markdown("### Context snapshot")
    st.json(context)

# Run MCP
if st.button("Run MCP and Get Recommendation"):
    result = mcp.run(features, context)
    st.markdown("## Recommendation")
    dec = result["decision"]
    if dec["action"] == "buy":
        st.success(f"BUY — confidence {dec['confidence']}")
    elif dec["action"] == "sell":
        st.error(f"SELL — confidence {dec['confidence']}")
    else:
        st.info(f"HOLD — confidence {dec['confidence']}")

    st.markdown("**Reasons / Explanation:**")
    for r in dec["reasons"]:
        st.write("- " + r)

    st.markdown("**Adjusted probabilities:**")
    st.write(pd.DataFrame([result["adjusted_probs"]]).T.rename(columns={0:"probability"}))

    st.markdown("**Raw model probabilities:**")
    st.write(pd.DataFrame([result["raw_model_probs"]]).T.rename(columns={0:"probability"}))

    st.markdown("**Export recommendation**")
    export = {
        "ticker": ticker,
        "features": features,
        "context": context,
        "decision": dec
    }
    st.download_button("Download JSON", data=json.dumps(export, indent=2), file_name=f"recommendation_{ticker}.json", mime="application/json")
else:
    st.markdown("Press **Run MCP and Get Recommendation** to generate a decision.")
    st.info("Tip: use the left sidebar to change the context (risk, exposure, sentiment).")
