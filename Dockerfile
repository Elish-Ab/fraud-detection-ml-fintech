# Use the official Python image from the Docker Hub
FROM python:3.10-slim
# Set the working directory in the container to /app
WORKDIR /app

# Install the dependencies specified in the requirements file
COPY requirements.txt .

# Copy the rest of the application code into the container
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Expose port 5000 to allow external access
EXPOSE 5000

# Specify the command to run the application
CMD ["python", "project/serve_model.py"]
