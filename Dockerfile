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

# Expose the port Railway assigns (default to 8000 if not set)
# Note: EXPOSE is informational; what matters is binding to the correct port in your app.
EXPOSE ${PORT:-8000}

# Copy the supervisor configuration file
COPY supervisord.conf /etc/supervisord.conf

# Run supervisord to start both FastAPI and Dash
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
