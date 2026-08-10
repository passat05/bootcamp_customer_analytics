FROM python:3.11-slim

WORKDIR /app

# 0. libgomp1 - required at runtime by lightgbm's compiled OpenMP extension.
# build-essential/gcc/g++ - required to compile scikit-survival's C/C++ extensions (e.g. ecos) at install time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 1. Cài đặt dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy toàn bộ thư mục app vào /app/app
# Điều này tạo ra cấu trúc: /app/app/api/main.py
COPY app/ /app/app/

# 3. Biến môi trường để Python tìm thấy module 'app'
ENV PYTHONPATH=/app

EXPOSE 8000

# 4. Chạy app bằng module path
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]