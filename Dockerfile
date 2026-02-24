FROM ubuntu:22.04

# Install Python + curl
RUN apt-get update && \
    apt-get install -y python3-pip curl

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync

# Copy project files
COPY train.py .
COPY src ./src
COPY app ./app

# Make run.sh executable
RUN chmod +x app/run.sh

# Expose port
EXPOSE 8000

# Start container
CMD ["bash", "-c", "./app/run.sh"]