#!/bin/bash
set -e

echo "🔹 Starting FinSight-NeuroQuant container..."

# -----------------------------
# Step 1: Generate seed data
# -----------------------------
echo "📊 Generating synthetic market data..."
python backend/seed_data.py

# -----------------------------
# Step 2: Train model
# -----------------------------
echo "🤖 Training investment model..."
python backend/train_model.py

# -----------------------------
# Step 3: Launch Streamlit app
# -----------------------------
echo "🚀 Launching Streamlit app..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0
