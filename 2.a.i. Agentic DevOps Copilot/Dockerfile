# Agentic DevOps Copilot — shared image for the FastAPI service and the Streamlit UI.
# docker-compose runs this same image as two services with different commands.
FROM python:3.11-slim

# Keep Python lean and unbuffered for clean container logs
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + mock data
COPY app/ ./app/
COPY data/ ./data/
COPY eval/ ./eval/
COPY streamlit_app.py .
COPY .streamlit/ ./.streamlit/

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Basic container healthcheck against the API
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
