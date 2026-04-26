FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY . .

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]