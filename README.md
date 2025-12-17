# A Mohammed Faazil

# FinSight — MCP (NeuroQuant)

🔗 **Live Demo:** https://finsight-mcp-production.up.railway.app

**FinSight** is an interview-ready FinTech project implementing a clear  
**MCP (Model • Context • Protocol)** architecture, enhanced with **NeuroQuant research concepts**.

The system provides **buy / hold / sell** recommendations using machine learning,
context-aware decision rules, and explainable reasoning.  
It is fully **Dockerized** and **deployed on the cloud** for live access.

---

## Architecture

### Model
- RandomForest classifier
- Predicts: **buy / hold / sell**
- Uses financial features such as momentum, volatility, valuation, sector signal, and liquidity

### Context
- User risk tolerance
- Portfolio exposure
- Market sentiment
- Investment horizon

### Protocol
- Business rules that adjust model outputs
- Risk-aware and auditable decisions

### NeuroQuant Integration
- Market regime detection using Wasserstein distance
- Diffusion-inspired smoothing for stable decisions
- Explainability layer for human-readable reasoning

---

## Project Contents

This repository contains:

- `backend/seed_data.py` — synthetic market data generator
- `backend/train_model.py` — trains the RandomForest model
- `backend/mcp.py` — base MCP logic
- `backend/orchestrator.py` — NeuroQuant MCP coordinator
- `backend/regime/` — regime detection logic
- `backend/policy/` — diffusion-based policy smoothing
- `backend/protocol/` — decision rules
- `backend/explainability/` — explanation generation
- `app.py` — Streamlit UI
- `Dockerfile` — Docker image definition
- `docker_entrypoint.sh` — automated pipeline runner

---

## Quick Start (Local)

```bash
# Windows (Git Bash)
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

python backend/seed_data.py
python backend/train_model.py
streamlit run app.py
