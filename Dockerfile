# Use the official Python image from Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install supervisor
RUN apt-get update && apt-get install -y supervisor

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose ports for both FastAPI (8000) and Dash (8050)
EXPOSE 8000 8050

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisord.conf

# Run supervisord to start both FastAPI and Dash
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
