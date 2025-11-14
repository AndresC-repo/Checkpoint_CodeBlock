FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create examples directory if it doesn't exist
RUN mkdir -p examples

# Set Python path to include the project root
ENV PYTHONPATH=/app

# Command to run the example
# CMD ["python", "examples/example1.py"]
# CMD ["python", "examples/example2.py"]
