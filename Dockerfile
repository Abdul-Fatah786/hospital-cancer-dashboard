FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch is much smaller than the default GPU build; install it first, then the rest
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# API on 8000, Streamlit on 8501 (see docker-compose.yml)
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
