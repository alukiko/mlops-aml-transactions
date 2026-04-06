# Образ HTTP API скоринга AML-транзакций
FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-api.txt .
RUN pip install --upgrade pip && pip install -r requirements-api.txt

COPY pyproject.toml README.md LICENSE ./
COPY mlops_aml_transactions ./mlops_aml_transactions/
RUN pip install --no-cache-dir .

# Модель вшивается в образ: перед сборкой положите models/model.pkl (и др. артефакты) в ./models/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "mlops_aml_transactions.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
