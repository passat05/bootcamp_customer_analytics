## MLflow Server Dockerfile (version 3.6.0)

FROM ghcr.io/mlflow/mlflow:v2.11.3

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    bash \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir psycopg2-binary boto3

COPY scripts/wait-for-it.sh /usr/local/bin/scripts/wait-for-it.sh
RUN chmod +x /usr/local/bin/scripts/wait-for-it.sh

WORKDIR /mlflow

RUN if ! id -u mlflow >/dev/null 2>&1; then \
    groupadd -r mlflow && useradd -r -g mlflow -m -d /home/mlflow -s /bin/bash mlflow || true; \
    fi

USER mlflow

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]