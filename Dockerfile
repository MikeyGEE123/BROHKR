# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Environment variables to prevent Python from writing .pyc files and buffering stdout/stderr.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to leverage Docker's layer caching
COPY requirements.txt .

# Install necessary Python packages
RUN pip install --upgrade pip && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; else echo "requirements.txt not found"; fi

# Copy the rest of the application code into the container
COPY . .

# Expose a port if your application listens on one (optional; adjust if needed)
EXPOSE 8000

# Default command to run your application
CMD ["python", "main.py"]
