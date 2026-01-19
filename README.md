## 📌 Business Context
Tôi đóng vai trò là **Data Scientist** tại một doanh nghiệp (Fintech/E-commerce), nơi mô hình tăng trưởng đang đối mặt với 3 thách thức lớn:
* **CAC (Cost Per Acquisition) tăng**
* **Churn cao ở nhóm khách hàng giá trị**
* **Chiến dịch Retention kém hiệu quả:** Các chương trình khuyến mãi được gửi đại trà (Mass Marketing), dẫn đến:
    * **Lãng phí ngân sách:** Gửi voucher cho người chắc chắn đi hoặc người chắc chắn ở lại.
    * **Over-treatment:** Gây phiền hà (spam) cho khách hàng, làm giảm trải nghiệm thương hiệu.

**Mục tiêu dự án:** Xác định chính xác **Top 20% khách hàng ưu tiên** để can thiệp, nhằm tối đa hóa tỷ lệ giữ chân và ROI của ngân sách Marketing.

---

## 🛠 Methodology
Dự án triển khai và so sánh 3 phương pháp tiếp cận từ truyền thống đến tiên tiến:

### 1. Phân khúc khách hàng

* Dựa trên RFM, thống kê được nhóm Hiberating and At Risk đang chiếm tỷ trọng khá cao, gần 40% khách hàng

<img width="226" height="129" alt="Screenshot 2026-01-19 at 00 41 48" src="https://github.com/user-attachments/assets/2252a575-f6c7-43ad-8125-4ef74223d8f6" />

* Sau khi gắn nhãn Churn dựa trên inactivity window (Inactivity window được chọn dưa trên ngưỡng trung bình từ 95%-98% thời gian mua hàng lặp lại), thống kê cho thấy nhóm khách churn hầu hết cũng đến từ nhóm Hiberating and At Risk

<img width="510" height="78" alt="Screenshot 2026-01-19 at 00 45 28" src="https://github.com/user-attachments/assets/db982f89-480e-48d1-8121-575be28e7845" />

# 2. Churn via ML Classification
Kết quả sau khi chạy các thuật toán học máy cho thấy các mô hình được có khả năng phân loại tốt với AUC > 91% và precision & recall > 80%. Trong đó mô hình Logistic Regression có nhỉnh hơn về Precision (89%) và đây cũng là yếu tố cân nhắc cho việc chọn mô hình xét đến việc giảm spam đến nhầm khách, gây lãng phí ngân sách.

<img width="318" height="78" alt="Screenshot 2026-01-19 at 00 49 55" src="https://github.com/user-attachments/assets/38a8561d-7842-4563-932c-90a503507638" />


### 3. Churn via BG-NBD
Sau khi chạy mô hình, ngưỡng phù hợp cho p_alive xác định khách churn là 0.8, khi đó churn label cho khách hàng đạt được độ khớp ~ 90% đối với cả khách churn và không churn

<img width="395" height="387" alt="Screenshot 2026-01-19 at 00 57 20" src="https://github.com/user-attachments/assets/b0b121d0-0709-4754-a0f2-3ccacd5a4f2c" />

## 4. Survival Analysis & CLV (Time-to-Event View)
Mô hình Weibull and CoxPH cho ra kết quả khá giống nhau cho Survival Curve
* Mô hình CoxPH

<img width="610" height="396" alt="Screenshot 2026-01-19 at 01 00 10" src="https://github.com/user-attachments/assets/9bb1b683-f551-40e7-974a-9095880a9333" />

* Mô hình Weibull

<img width="621" height="387" alt="Screenshot 2026-01-19 at 01 01 43" src="https://github.com/user-attachments/assets/3ebf60d1-2e43-4797-afa8-70ec5bbea7b2" />

Kết quả CLV theo phương pháp BG-NBD + Gamma–Gamma: 50% CLV 3 tháng tới rơi vào giá trị 573

<img width="143" height="120" alt="Screenshot 2026-01-19 at 01 19 51" src="https://github.com/user-attachments/assets/03d9924e-588c-42ba-ad7a-f53fd798ecc7" />

Kết quả CLV theo Survival Analysis + Gamma–Gamma: 50% CLV còn lại rơi vào giá trị 463

<img width="146" height="129" alt="Screenshot 2026-01-19 at 01 15 54" src="https://github.com/user-attachments/assets/a4a67786-88b9-4cb6-ad61-4763012ac888" />

---

## 📊 Decision Matrix: Choosing the Right Strategy

| Tiêu chí | 1. Classification | 2. BG-NBD | 3. Survival-based CLV |
| :--- | :--- | :--- | :--- |
| **Số ngày giao dịch gần nhất** | 177 | 187 | 6 |
| **Giá trị giao dịch trong năm** | 702 | 678 | 533 |

💡 Key Insights  

Khách hàng ưu tiên ở chiến lược 1,2 là những khách hàng đã giao dịch lần cuối khá lâu gần nửa năm nhưng giá trị giao dịch cao.
Trong khi khách hàng ưu tiên ở chiến lược 3 lại là những khách hàng mới gần giao dịch và giao dịch ở mức trung bình.
Mặc dù khách hàng trong tập chiến lược 1,2 có thể mang lại giá trị nếu đánh thức được tuy nhiên mức độ khả thi kho vì đã ngưng giao dịch quá lâu.
Do đó ưu tiên nhóm khách ở nhóm 3 sẽ có tính khả thi cao hơn.

---

## 📂 Project Structure
* `data/`: Dữ liệu RFM-T thô.
* `notebooks/`: 
    * `customer_analysis.ipynb`
