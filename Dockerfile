FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860

WORKDIR /home/user/app

# Copy and install requirements first (Docker layer cache)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
# NOTE: qdrant_hindi_benchmark/ is intentionally NOT copied — the app
# reads from Qdrant Cloud via QDRANT_URL and QDRANT_API env vars.
COPY --chown=user . .

EXPOSE 7860

CMD ["python", "api.py"]
