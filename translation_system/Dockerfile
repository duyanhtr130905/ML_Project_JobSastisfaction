FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download MarianMT model (pre-download for faster startup)
RUN python -c "from transformers import MarianMTModel, MarianTokenizer; \
    MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-vi'); \
    MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-vi')"

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p /data/translation_memory

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Run the application
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
