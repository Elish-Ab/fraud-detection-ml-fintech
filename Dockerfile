# Use the official Python image from Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose ports for both FastAPI (8000) and Dash (8050)
EXPOSE 8000 8050

# Run both FastAPI and Dash in the same container
CMD uvicorn project.serve_model:app --host 0.0.0.0 --port 8000 & python project/dashboard.py
