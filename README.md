# 🏥 Healthcare AI Agent: Decision-Support Assistant

Dự án này xây dựng một hệ thống hỗ trợ quyết định y tế tích hợp AI, giúp người dùng chẩn đoán triệu chứng sơ bộ, tìm kiếm bác sĩ chuyên khoa và dự báo chi phí bảo hiểm dựa trên mô hình Machine Learning.

## ✨ Tính năng chính
- **Chẩn đoán triệu chứng AI:** Sử dụng **FAISS** và **Sentence Transformers** để phân tích triệu chứng và đưa ra chẩn đoán sơ bộ.
- **Ước tính bảo hiểm:** Mô hình **Random Forest Regressor** dự báo chi phí y tế dựa trên các chỉ số cá nhân (BMI, độ tuổi, vùng miền).
- **Tìm kiếm bác sĩ:** Tích hợp bản đồ trực quan hỗ trợ tìm kiếm bác sĩ chuyên khoa tại các khu vực (đã cập nhật dữ liệu tại TP.HCM & Cần Thơ).
- **Bộ nhớ bệnh nhân:** Lưu trữ lịch sử tương tác để tăng độ chính xác cho các lần chẩn đoán sau.

## 🛠 Tech Stack
- **Ngôn ngữ:** Python 3.13+
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **NLP & Search:** FAISS, Sentence-Transformers
- **Giao diện:** Gradio (Web-based UI)

## 🚀 Hướng dẫn cài đặt
1. Clone repository:
   ```bash
   git clone [https://github.com/TranGiaHy/Healthcare_AI_Agent.git](https://github.com/TranGiaHy/Healthcare_AI_Agent.git)
   cd Healthcare_AI_Agent
