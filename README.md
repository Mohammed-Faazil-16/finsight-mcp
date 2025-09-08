# FinSight — MCP 

**FinSight** is an interview-ready, offline FinTech demo implementing a clear **MCP (Model • Context • Protocol)** architecture:
- **Model**: RandomForest classifier that scores assets (buy / hold / sell).
- **Context**: User risk profile, portfolio exposure, market sentiment, horizon.
- **Protocol**: Business rules that adjust model output and produce auditable decisions.

This repo contains:
- `backend/seed_data.py` — synthetic seed dataset generator.
- `backend/train_model.py` — trains and saves the RandomForest model.
- `backend/mcp.py` — MCP orchestrator (Model→Context→Protocol).
- `app.py` — Streamlit UI to demo the MCP pipeline locally.

## Quick start

```bash
# Windows (Git Bash)
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

python backend/seed_data.py
python backend/train_model.py
streamlit run app.py
