# Use a lightweight Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files from the service directory
COPY src/api/pyproject.toml ./
# We don't have a lock file yet, uv will create one
# COPY src/api/uv.lock ./ 

# Install dependencies
RUN uv sync --no-install-project

# Copy source code (we need the whole src because of cross-imports)
COPY src ./src

# Expose FastAPI port
EXPOSE 8000

# Run the API
CMD ["uv", "run", "python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
