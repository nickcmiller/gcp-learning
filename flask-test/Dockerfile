# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Flask application code into the container
COPY app.py .

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
PORT 8080

# Define the command to run the Flask application
CMD ["python", "app.py"]