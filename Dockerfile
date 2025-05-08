# Multi-stage build

# Backend (FastAPI)
FROM python:3.11-slim AS backend
WORKDIR /backend
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend (Streamlit)
FROM python:3.11-slim AS frontend
WORKDIR /frontend
COPY frontend/requirements.txt .
RUN pip install -r requirements.txt
COPY frontend/ .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
