# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only necessary files first to optimize caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container
COPY . .

# Expose port 5000 for the Flask API
EXPOSE 5000

# Run the model serving script
CMD ["python", "project/serve_model.py"]
