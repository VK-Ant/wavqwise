FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml .
COPY wavqwise/ wavqwise/
COPY demos/ demos/

RUN pip install --no-cache-dir -e ".[traditional,viz]"

EXPOSE 8000 8888

CMD ["python", "-m", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
