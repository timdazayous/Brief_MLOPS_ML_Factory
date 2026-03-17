# Use a lightweight Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy project files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-install-project

# Copy source code and data
COPY src ./src
COPY data ./data

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["uv", "run", "streamlit", "run", "src/front/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
