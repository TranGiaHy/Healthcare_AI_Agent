# ⚕️ Healthcare AI Agent: Decision-Support Assistant

[![Python Version](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Một hệ thống hỗ trợ quyết định y tế (Medical Decision Support System) thông minh, tích hợp học máy và xử lý ngôn ngữ tự nhiên để hỗ trợ chẩn đoán sơ bộ, dự báo chi phí bảo hiểm và tối ưu hóa quy trình tìm kiếm bác sĩ.

---

## 🌟 Tính năng nổi bật

* **🤖 Chẩn đoán triệu chứng thông minh:** Sử dụng **Sentence-Transformers** và **FAISS** để thực hiện tìm kiếm ngữ nghĩa (Semantic Search) trên cơ sở dữ liệu triệu chứng, đưa ra chẩn đoán sơ bộ với độ tin cậy cao.
* **📊 Dự báo bảo hiểm (Insurance ML):** Tích hợp mô hình **Random Forest Regressor** để ước tính chi phí y tế dựa trên các chỉ số sinh trắc học (BMI, tuổi, vùng miền, tình trạng hút thuốc).
* **📍 Tìm kiếm bác sĩ thông minh:** Tự động đề xuất bác sĩ chuyên khoa dựa trên chẩn đoán và vị trí địa lý, tích hợp bản đồ trực quan (TP.HCM & Cần Thơ).
* **🧠 Bộ nhớ bệnh nhân (Patient Memory):** Hệ thống lưu trữ lịch sử tương tác cá nhân hóa, giúp AI hiểu rõ hơn về tiến trình sức khỏe của bệnh nhân qua các lần thăm khám.
* **🎨 Giao diện hiện đại (Premium UI/UX):** Giao diện Web được tối ưu hóa với Gradio, mang phong cách y tế chuyên nghiệp, hỗ trợ trải nghiệm người dùng mượt mà.

---

## 🛠 Tech Stack

* **Core Engine:** Python 3.13+
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **NLP & Vector Search:** FAISS, Sentence-Transformers (All-MiniLM-L6-v2)
* **User Interface:** Gradio (Custom CSS/HTML)
* **Data Management:** JSON-based persistent memory

---

## 📂 Cấu trúc thư mục

```text
Healthcare_AI_Agent/
├── core/                   # Engine xử lý trung tâm
│   ├── diagnostics.py      # Logic chẩn đoán NLP & FAISS
│   ├── insurance.py        # Model ML dự báo chi phí
│   ├── doctors.py          # Quản lý dữ liệu và bản đồ bác sĩ
│   ├── memory.py           # Xử lý lưu trữ lịch sử bệnh nhân
│   └── explainer.py        # Giải thích kết quả bằng AI
├── data/                   # Tập dữ liệu (Symptoms, Doctors, v.v.)
├── ui/                     
│   └── app.py              # File chạy chính (Gradio UI)
├── requirements.txt        # Danh sách thư viện cài đặt
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt
1. Clone repository:

```bash
git clone [https://github.com/TranGiaHy/Healthcare_AI_Agent.git](https://github.com/TranGiaHy/Healthcare_AI_Agent.git)
cd Healthcare_AI_Agent
```

2. Cài đặt môi trường:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Khởi chạy ứng dụng:

```bash
python -m ui.app
```

---

## 👤 Tác giả
Trần Gia Hy - Sinh viên năm cuối chuyên ngành Data Science, HUFLIT.

Mục tiêu: Phát triển các giải pháp AI ứng dụng thực tiễn trong lĩnh vực y tế và đời sống.

Liên hệ: [trangiahy31082004@gmail.com]

---

## 📜 Disclaimer
Dự án này được xây dựng cho mục đích học thuật và nghiên cứu. Các chẩn đoán từ AI chỉ mang tính chất tham khảo, không thay thế cho lời khuyên y tế chuyên nghiệp từ bác sĩ.

