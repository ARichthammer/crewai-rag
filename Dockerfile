# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Code Engine uses PORT env variable (default 8080)
ENV PORT=8080
EXPOSE 8080

# Run with dynamic port from Code Engine
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
