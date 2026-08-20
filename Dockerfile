FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (required by Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR /home/user/app

# Copy requirements first for better Docker layer caching
COPY --chown=user requirements.txt .

# Upgrade pip and install all Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
# NOTE: The local qdrant_hindi_benchmark/ DB is NOT included.
# At runtime, the app connects to Qdrant Cloud via QDRANT_URL + QDRANT_API env vars.
COPY --chown=user . .

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Run the Flask app
CMD ["python", "api.py"]
