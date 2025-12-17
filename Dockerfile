# -----------------------------
# Base image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Copy requirements & install
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy project files
# -----------------------------
COPY . .

# -----------------------------
# Expose Streamlit port
# -----------------------------
EXPOSE 8501

# -----------------------------
# Run entrypoint script
# -----------------------------
CMD ["bash", "docker_entrypoint.sh"]
