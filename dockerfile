# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the application files to the container
COPY flask_webpage /app/flask_webpage

# Copy the requirements file to the container
COPY flask_webpage/requirements.txt /app/requirements.txt

# Copy the model file to the container
COPY flask_webpage/license_plate_recognition_model.h5 /app/license_plate_recognition_model.h5

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port the app runs on
EXPOSE 5000

# Define the command to run the app
CMD ["python", "/app/flask_webpage/main.py"]
