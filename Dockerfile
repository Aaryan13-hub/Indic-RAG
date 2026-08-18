FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up user to avoid running as root (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY --chown=user:user . /app

# Switch to the non-root user
USER user

# Set environment variables
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV PORT=7860

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Run the Flask app
CMD ["python", "api.py"]
