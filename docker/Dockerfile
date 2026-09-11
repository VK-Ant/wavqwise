FROM python:3.12-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install wavqwise
COPY pyproject.toml README.md LICENSE ./
COPY wavqwise/ wavqwise/
RUN pip install --no-cache-dir -e ".[traditional,ml,viz,database]"

# Copy demos
COPY demos/ demos/

# Generate sample data
RUN cd demos/sample_data && python generate_sample_data.py

EXPOSE 8000 8888

# Default: run forecasting demo
CMD ["python", "demos/demo_forecasting.py"]
