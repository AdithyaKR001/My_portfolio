# Incident-RAG — shared image for ingest, the FastAPI service, and the Streamlit UIs.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag/ ./rag/
COPY eval/ ./eval/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY dashboard.py chat_ui.py ./

RUN useradd --create-home appuser && mkdir -p /app/data/chroma && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
