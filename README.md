# Customer Analytics Platform

**Hệ thống MLOps end-to-end cho churn, lifetime value và survival modeling của khách hàng** — từ dữ liệu giao dịch thô đến một API dự đoán được triển khai và giám sát.

---

## Bài toán kinh doanh

Vai trò: Data Scientist tại một doanh nghiệp Fintech/E-commerce đang đối mặt ba vấn đề:

1. **CAC tăng.**
2. **Churn cao, tập trung ở nhóm khách hàng giá trị.**
3. **Retention bị mass-marketing** — gửi voucher đại trà, lãng phí ngân sách cho người chắc chắn ở lại/rời đi dù có can thiệp hay không, và gây phiền hà (over-treatment) cho phần còn lại.

**Mục tiêu:** xác định **top 20% khách hàng ưu tiên** để can thiệp, tối đa hoá tỷ lệ giữ chân và ROI ngân sách marketing.

---

## Giải pháp

| Bước | Nội dung |
|---|---|
| **1. EDA định lượng vấn đề** | [`notebooks/01_eda_business_insights.ipynb`](notebooks/01_eda_business_insights.ipynb) — đo mức độ tập trung doanh thu, phân khúc RFM, và định lượng chính xác mức lãng phí của mass-marketing bằng số liệu, không phải phỏng đoán. |
| **2. Propensity model chuẩn chỉnh cho churn** | LightGBM huấn luyện theo **stacked cohorts + out-of-time validation**, đánh giá bằng decile/lift, precision@20%/capture@20%, calibration, SHAP — theo đúng phương pháp luận Propensity Model. Chi tiết: [`docs/model_card_churn.md`](docs/model_card_churn.md). |
| **3. So sánh 3 chiến lược & mô phỏng chiến dịch** | [`notebooks/02_propensity_targeting_campaign.ipynb`](notebooks/02_propensity_targeting_campaign.ipynb) — xây 3 mô hình xác định top 20% (phân loại LightGBM, BG-NBD, Survival + Gamma-Gamma), **chọn chiến lược dựa trên dữ liệu thực tế** (không phải giả định), rồi mô phỏng ROI có holdout/control-group so với mass-marketing và random 20%. |
| **4. Vận hành MLOps** | Toàn bộ pipeline train → evaluate → promote → monitor chạy qua Airflow + MLflow + FastAPI, có cổng promote an toàn (so sánh model mới và model cũ trên cùng một bộ dữ liệu trước khi quyết định thay thế). |

---

## Kết quả kinh doanh

**Từ EDA** (dữ liệu thật, không mô phỏng):
- **Top 20% khách hàng tạo ra 70% doanh thu** — cơ sở định lượng cho khái niệm "khách hàng giá trị".
- **~60% lượt tiếp cận của một chiến dịch đại trà bị lãng phí** — rơi vào nhóm chắc chắn ở lại hoặc chắc chắn không phản hồi.

**Từ propensity model** (OOT — chưa từng thấy lúc train):
- ROC-AUC = **0.79**, decile-1 lift = **1.65x**, top 20% khách hàng ưu tiên bắt được **32.9%** tổng số khách sẽ churn thật — gấp 1.65 lần ngẫu nhiên.

**Từ so sánh 3 chiến lược xác định top 20%** (kiểm chứng bằng dữ liệu thực tế 30 ngày sau snapshot):

| Chiến lược | Độ chính xác trong nhóm (thực sự churn) | % tổng số khách churn bắt được |
|---|---|---|
| Xác suất churn cao (phân loại) | 79.7% | 33.7% |
| P(alive) thấp (BG-NBD) | 64.4% | 27.2% |
| **CLV cao × rủi ro cao (Survival) — thắng cuộc** | **96.7%** | **40.9%** |

**Từ mô phỏng chiến dịch** (dùng chiến lược thắng cuộc, holdout/control-group, giả định thận trọng):

| Chiến lược | Chi phí | ROI |
|---|---|---|
| Ngẫu nhiên 20% | 1/5 chi phí mass | ~72% |
| Mass marketing (100%) | Cao nhất | ~83% |
| **Survival-based, top 20% (đề xuất)** | 1/5 chi phí mass | **~96%** |

---

## Kiến trúc & công nghệ

| Nhóm | Công nghệ |
|---|---|
| Ngôn ngữ / package | Python 3.11, `uv` |
| Modeling | LightGBM, `lifelines`, `lifetimes`, SHAP |
| Serving | FastAPI |
| Tracking / registry | MLflow (Postgres backend, MinIO artifact store) |
| Orchestration | Apache Airflow (CeleryExecutor) |
| Giám sát | Prometheus, Grafana |
| Hạ tầng | Docker Compose (12 container) |

**Vài lựa chọn kỹ thuật đáng chú ý:**
- Một module `feature_engineering.py` duy nhất dùng chung cho training lẫn serving — loại bỏ hoàn toàn training-serving skew.
- Model registry theo alias (`candidate`/`prod`), không load file `.pkl` tĩnh — promote là một lệnh, không cần redeploy.
- Mọi feature đều tuân thủ point-in-time (chỉ dùng dữ liệu ≤ snapshot_date) — kỷ luật này từng giúp phát hiện một sự cố leakage thật trong model production, chi tiết tại [`docs/model_card_churn.md`](docs/model_card_churn.md).

Chi tiết đầy đủ (lý do thiết kế, xử lý lỗi, khả năng mở rộng...) nằm trong code và docstring của từng file trong `src/` và `scripts/`.

---

## Cấu trúc repository

```
src/            # Feature engineering, schema, inference — nguồn định nghĩa duy nhất
scripts/        # CLI: train (--stacked), evaluate (business metrics), promote (fresh-holdout), monitor drift
notebooks/      # 01_eda_business_insights, 02_propensity_targeting_campaign
docs/           # model_card_churn.md — phương pháp luận + sự cố đã điều tra
airflow/        # 3 DAG: simulate_data → feature_transform → train_and_promote
main.py         # FastAPI serving
docker-compose.yml
```

---

## Bắt đầu nhanh

```bash
docker compose up -d
```

| Service | URL |
|---|---|
| FastAPI docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5001 |
| Airflow | http://localhost:8080 (`airflow`/`airflow`) |

**Chạy đầy đủ phương pháp luận propensity model:**
```bash
python scripts/train_model.py --stacked data/customers.csv data/transactions.csv
python scripts/evaluate_business_metrics.py data/transactions.csv candidate
python scripts/promote_model.py Churn_model data/transactions.csv   # bắt buộc kèm transactions.csv — xem model card
python scripts/monitor_drift.py data/processed/churn_dataset.csv data/transactions.csv prod
```
