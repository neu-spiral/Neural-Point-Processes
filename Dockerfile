# Using an official Python runtime as a parent image
FROM python:3.8.17-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install system dependencies required for compiling certain Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file, to cache the installed
# dependencies unless the requirements file itself changes
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project into the container at /app
COPY . /app

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Start bash to run experiments
CMD ["/bin/bash"]

