# Cố định Airflow + Python version
FROM apache/airflow:2.9.2-python3.10

# ======================
# System dependencies (chạy quyền root)
# ======================
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        curl \
        libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
#   ^^^^^^^^^^^
#   libgomp1 bắt buộc cho LightGBM trên Linux

# ======================
# Python dependencies (chạy quyền airflow)
# ======================
USER airflow

COPY --chown=airflow:root requirements.txt /requirements.txt

RUN pip install --no-cache-dir -r /requirements.txt \
    apache-airflow-providers-amazon \
    dvc[s3]

# ======================
# Verify
# ======================
RUN airflow version && \
    python -c "import mlflow; print('MLflow:', mlflow.__version__)" && \
    python -c "import lightgbm; print('LightGBM:', lightgbm.__version__)" && \
    python -c "import lifelines; print('Lifelines:', lifelines.__version__)"