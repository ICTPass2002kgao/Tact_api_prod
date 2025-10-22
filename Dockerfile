# Use an official Python runtime as a parent image (Debian based)
FROM python:3.13-slim

# Set environment variables for non-interactive install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1

# Install libGL.so.1 dependency (essential for OpenCV)
RUN apt-get update && \
    apt-get install -y libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python packages
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Use the standard port 8080 for web services
ENV PORT 8080

# Command to run the Django application with Gunicorn
# Replace 'tact_api.wsgi' with your actual WSGI path if different
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "tact_api.wsgi"]