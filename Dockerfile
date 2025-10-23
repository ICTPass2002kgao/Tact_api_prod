# Use an official Python runtime as a parent image (Debian based)
# Using python:3.11-slim for better compatibility with ML libraries
FROM python:3.11-slim

# Set environment variables for non-interactive install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1

# === BEGIN REQUIRED SYSTEM DEPENDENCIES ===
# Install C++ build tools (cmake, build-essential) required for dlib compilation,
# and runtime libraries (libgl1, python3-dev) for dlib/OpenCV.
RUN apt-get update && \
    apt-get install -y \
    cmake \
    build-essential \
    libgl1 \
    libx11-dev \
    libgtk-3-dev \
    python3-dev \
    && \
    rm -rf /var/lib/apt/lists/*
# === END REQUIRED SYSTEM DEPENDENCIES ===

# Set the working directory
WORKDIR /app

# Copy the requirements file and install Python packages
# NOTE: The pip install should now succeed because the system dependencies are installed.
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Use the standard port 8080 for web services
ENV PORT 8080

# Command to run the Django application with Gunicorn
# Replace 'tact_api.wsgi' with your actual WSGI path if different
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "tact_api.wsgi"]