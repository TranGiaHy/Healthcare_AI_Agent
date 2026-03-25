import gradio as gr
import os
import sys

# Thiết lập đường dẫn thư mục gốc để import các module từ core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.diagnostics import SymptomDiagnosticEngine
from core.doctors import find_doctors, create_doctor_map
from core.insurance import InsuranceEstimator
from core.memory import generate_patient_id, update_patient_memory, get_patient_history
from core.explainer import AIExplainer

# Khởi tạo các engine xử lý chính của ứng dụng
diagnostic_engine = SymptomDiagnosticEngine()
insurance_estimator = InsuranceEstimator()
ai_explainer = AIExplainer()

# =====================================================================
# PHẦN 1: LOGIC XỬ LÝ BACKEND
# =====================================================================

# Tính toán chỉ số BMI dựa trên chiều cao (cm) và cân nặng (kg)
def calculate_bmi(height_cm, weight_kg):
    height_m = float(height_cm) / 100.0
    if height_m <= 0:
        raise ValueError("Height must be greater than zero.")
    bmi = float(weight_kg) / (height_m ** 2)
    return round(bmi, 2)

# Định dạng danh sách bác sĩ thành chuỗi văn bản hiển thị
def format_doctors(doctors):
    if not doctors:
        return "No doctors found."
    lines = []
    for doc in doctors:
        lines.append(
            f"{doc['name']} | {doc['specialty']} | {doc['city']} | Rating: {doc['rating']}"
        )
    return "\n".join(lines)

# Xây dựng cấu trúc hiển thị lịch sử khám bệnh của bệnh nhân
def format_history(history):
    if not history:
        return "No previous patient history found."

    visits = history.get("history", [])
    if not visits:
        return "No previous patient history found."

    lines = []
    lines.append("Patient Visit Timeline")
    lines.append("-" * 40)

    for idx, visit in enumerate(visits, start=1):
        diag = visit.get("diagnosis", {})
        top = diag.get("top_diagnosis", {}) if isinstance(diag, dict) else {}
        demographics = history.get("demographics", {})

        lines.append(f"Visit {idx}")
        lines.append(f"Timestamp: {visit.get('timestamp', 'Unknown')}")
        lines.append(f"Symptoms: {visit.get('symptoms', 'Unknown')}")
        lines.append(f"Health insight: {top.get('diagnosis', 'Unknown')}")
        lines.append(f"Recommended specialty: {top.get('specialty', 'Unknown')}")
        lines.append(f"Insurance estimate: {visit.get('insurance_price', 'Unknown')}")
        if demographics.get("bmi") is not None:
            lines.append(f"Stored BMI: {demographics.get('bmi')}")
        lines.append("-" * 40)

    return "\n".join(lines)

# Trích xuất và định dạng tóm tắt kết quả chẩn đoán
def format_summary(result):
    top = result.get("top_diagnosis")
    if not top:
        return "No health insight found."

    summary_lines = [
        f"Health Insight: {top.get('diagnosis', 'Unknown')}",
        f"Recommended Specialty: {top.get('specialty', 'Unknown')}",
        f"Urgency Level: {top.get('triage_level', 'Unknown')}",
    ]

    if result.get("calculated_bmi") is not None:
        summary_lines.append(f"Calculated BMI: {result.get('calculated_bmi')}")

    return "\n".join(summary_lines)

# Tạo báo cáo chẩn đoán chi tiết với đầy đủ thông số
def format_detailed_analysis(result):
    top = result.get("top_diagnosis")
    if not top:
        return "No detailed analysis found."

    lines = []
    lines.append("Structured Health Analysis")
    lines.append("-" * 40)
    lines.append(f"Matched symptom pattern: {top.get('symptom_pattern', 'Unknown')}")
    lines.append(f"Health insight: {top.get('diagnosis', 'Unknown')}")
    lines.append(f"Recommended specialty: {top.get('specialty', 'Unknown')}")
    lines.append(f"Urgency level: {top.get('triage_level', 'Unknown')}")
    lines.append(f"Suggested tests: {top.get('recommended_tests', 'Unknown')}")
    lines.append(f"Advice: {top.get('advice', 'Unknown')}")
    lines.append(f"Similarity confidence: {top.get('confidence', 0):.4f}")

    if result.get("calculated_bmi") is not None:
        lines.append(f"Calculated BMI: {result.get('calculated_bmi')}")

    if result.get("history_summary"):
        lines.append("")
        lines.append(f"History summary: {result['history_summary']}")

    if result.get("safety_disclaimer"):
        lines.append("")
        lines.append(result["safety_disclaimer"])

    return "\n".join(lines)

# Hàm xử lý chính: Tiếp nhận input, gọi các AI Engine và trả về kết quả
def analyze(name, email, age, sex, height_cm, weight_kg, children, smoker, region, symptoms, city, use_ai):
    try:
        if not name.strip() or not email.strip() or not symptoms.strip():
            msg = "Please fill in name, email, and symptoms."
            return msg, msg, msg, msg, msg, msg, "<p>No map available.</p>"

        patient_id = generate_patient_id(name, email)

        result = diagnostic_engine.diagnose(symptoms, patient_id)
        top = result.get("top_diagnosis")

        if not top:
            msg = "No health insight could be generated."
            return msg, msg, msg, msg, msg, msg, "<p>No map available.</p>"

        doctors = find_doctors(top.get("specialty", ""), city)

        bmi = calculate_bmi(height_cm, weight_kg)
        result["calculated_bmi"] = bmi

        insurance_price, insurance_text = insurance_estimator.estimate(
            patient_id=patient_id,
            age=int(age),
            sex=sex,
            bmi=float(bmi),
            children=int(children),
            smoker=smoker,
            region=region,
        )

        if use_ai and not ai_explainer.model_loaded:
            ai_explainer.enable_local_model()

        payload = {
            "top_diagnosis": top,
            "doctors": doctors,
            "insurance_text": insurance_text,
            "history_summary": result.get("history_summary", ""),
        }

        ai_text = ai_explainer.explain(payload, use_ai=use_ai)

        demographics = {
            "name": name,
            "email": email,
            "age": age,
            "sex": sex,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region,
            "city": city,
        }

        update_patient_memory(
            patient_id=patient_id,
            demographics=demographics,
            symptoms=symptoms,
            diagnosis_result=result,
            insurance_price=insurance_price,
            doctors=doctors,
        )

        history = get_patient_history(patient_id)

        detailed_analysis = format_detailed_analysis(result)
        summary = format_summary(result)
        doctor_text = format_doctors(doctors)
        history_text = format_history(history)
        doctor_map_html = create_doctor_map(doctors)

        return (
            detailed_analysis,
            insurance_text,
            summary,
            ai_text,
            history_text,
            doctor_text,
            doctor_map_html,
        )

    except Exception as e:
        msg = f"Error during analysis: {e}"
        return msg, msg, msg, msg, msg, msg, "<p>No map available.</p>"

# Truy xuất và hiển thị dữ liệu lịch sử bệnh nhân
def show_history(name, email):
    try:
        if not name.strip() or not email.strip():
            return "Please enter both name and email."
        patient_id = generate_patient_id(name, email)
        history = get_patient_history(patient_id)
        return format_history(history)
    except Exception as e:
        return f"Error while loading history: {e}"

# =====================================================================
# PHẦN 2: HTML VÀ CSS CỦA GIAO DIỆN
# =====================================================================

# Cấu trúc HTML thanh điều hướng Header
HEADER_HTML = """
<div class="medvill-header">
    <div class="medvill-logo">
        <span style="font-size: 32px;">⚕️</span> 
        <span style="color: #003399; font-weight: 800; font-size: 26px;">Healthcare AI Agent</span>
    </div>
    <div class="medvill-nav">
        <span>HOME</span> <span>ABOUT</span> <span>DEPARTMENTS</span> 
        <span>TIMETABLE</span> <span>PAGES</span> <span>BLOG</span> <span>CONTACT</span>
    </div>
    <div class="medvill-icons">
        <span>📅</span> <span>🔔</span> <span>🔍</span> <span>☰</span>
    </div>
</div>
"""

# Cấu trúc HTML phần Hero Section (Banner chính)
HERO_HTML = """
<div class="medvill-hero">
    <h4 style="color: #93c5fd !important;">WELCOME TO HEALTHCARE AI</h4>
    <h1 style="color: #ffffff !important;">MEDICAL CENTER</h1>
    <p style="color: #e2e8f0 !important;">We provide the best medical services in your city with modern, sophisticated equipment and highly experienced doctors.</p>
    <button class="hero-btn">DISCOVER MORE</button>
</div>
"""

# Cấu trúc HTML các thẻ thông tin liên hệ và dịch vụ
INFO_CARDS_HTML = """
<div class="info-cards-wrapper">
    <div class="info-card"><div class="info-card-icon">📱</div><div class="info-card-text"><h5>Emergency Cases</h5><p>+84798766186</p></div></div>
    <div class="info-card"><div class="info-card-icon">✉️</div><div class="info-card-text"><h5>Email Us</h5><p>trangiahy31082004@healthcare.com</p></div></div>
    <div class="info-card"><div class="info-card-icon">📅</div><div class="info-card-text"><h5>Working Hours</h5><p>View Timetable</p></div></div>
    <div class="info-card"><div class="info-card-icon">⏱️</div><div class="info-card-text"><h5>24/7 Service</h5><p>Always Open</p></div></div>
</div>
"""

# Cấu trúc HTML tiêu đề phần Form tương tác
MAIN_APP_TITLE_HTML = """
<div class="app-title-container">
    <h2 class="app-section-title">AI MEDICAL SERVICES</h2>
    <p class="app-section-subtitle">Experience the future of healthcare. Enter your details below to receive instant, AI-powered preliminary diagnostics and insurance estimates.</p>
</div>
"""

# Cấu trúc HTML phần Footer chứa các số liệu thống kê
FOOTER_HTML = """
<div class="medvill-footer">
    <div class="stat-box"><div class="stat-icon">🏥</div><div class="stat-text"><h3 style="color: #ffffff !important;">245k</h3><p style="color: #94a3b8 !important;">Happy Patients</p></div></div>
    <div class="stat-box"><div class="stat-icon">📋</div><div class="stat-text"><h3 style="color: #ffffff !important;">30+</h3><p style="color: #94a3b8 !important;">Departments</p></div></div>
    <div class="stat-box"><div class="stat-icon">👨</div><div class="stat-text"><h3 style="color: #ffffff !important;">122+</h3><p style="color: #94a3b8 !important;">Expert Doctors</p></div></div>
    <div class="stat-box"><div class="stat-icon">🏢</div><div class="stat-text"><h3 style="color: #ffffff !important;">100+</h3><p style="color: #94a3b8 !important;">Branches</p></div></div>
</div>
"""

# Biến chứa toàn bộ CSS tùy chỉnh cho giao diện Gradio
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Thiết lập nền trang màu xám rõ rệt */
body, html, .gradio-container { 
    background-color: #e5e7eb !important; /* Màu xám bê tông nhạt, đậm hơn trước */
    font-family: 'Plus Jakarta Sans', sans-serif !important; 
    max-width: 100% !important; 
    padding: 0 !important; 
    margin: 0 !important; 
    overflow-x: hidden; 
}

.gradio-container > .main, .gradio-container > .main > .wrap { padding: 0 !important; margin: 0 !important; max-width: 100% !important; }
footer { display: none !important; }

/* Định dạng Header và Logo */
.medvill-header { display: flex !important; justify-content: space-between !important; align-items: center !important; padding: 15px 40px !important; background: rgba(255, 255, 255, 0.95) !important; backdrop-filter: blur(10px) !important; -webkit-backdrop-filter: blur(10px) !important; border-radius: 50px !important; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1) !important; position: absolute !important; top: 25px !important; left: 5% !important; width: 90% !important; z-index: 1000 !important; box-sizing: border-box !important; }
.medvill-logo { display: flex; align-items: center; gap: 10px; letter-spacing: 0.5px; }
.medvill-nav { display: flex; gap: 35px; font-size: 14px; font-weight: 800; color: #003399; text-transform: uppercase; }
.medvill-nav span { color: #003399 !important; cursor: pointer; transition: color 0.3s ease; }
.medvill-nav span:hover { color: #ea580c !important; }
.medvill-icons { display: flex; gap: 25px; color: #003399; font-size: 18px; cursor: pointer; }

/* Định dạng khối Banner Hero */
.medvill-hero { 
    background: linear-gradient(to right, rgba(0, 20, 90, 0.85), rgba(0, 51, 153, 0.65)), url('https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80') center/cover; 
    padding: 200px 5% 200px !important; 
    text-align: center; 
    color: white; 
    box-sizing: border-box !important; 
    margin-top: -50px !important;
    
    width: 100vw !important; 
    position: relative !important; 
    left: 50% !important; 
    right: 50% !important; 
    margin-left: -50vw !important; 
    margin-right: -50vw !important; 
}

.medvill-hero h4 { font-size: 14px; letter-spacing: 4px; margin-bottom: 20px; font-weight: 700; color: #93c5fd !important; }
.medvill-hero h1 { font-size: 76px; font-weight: 800; margin: 0 0 25px; line-height: 1.1; color: #ffffff !important; }
.medvill-hero p { font-size: 18px; max-width: 650px; margin: 0 auto 40px; color: #e2e8f0 !important; }
.hero-btn { background: #ffffff; color: #003399; padding: 18px 40px; font-size: 15px; font-weight: 700; border-radius: 50px; border: none; cursor: pointer; transition: all 0.3s ease; }

/* Định dạng thẻ thông tin dưới Hero */
.info-cards-wrapper { display: flex; justify-content: center; gap: 25px; margin-top: -70px; padding: 0 8%; position: relative; z-index: 10; }
.info-card { background: #ffffff !important; padding: 35px 25px; border-radius: 16px; color: #0f172a; display: flex; align-items: center; gap: 15px; flex: 1; box-shadow: 0 20px 40px rgba(0,0,0,0.08); border: 2px solid transparent !important; cursor: pointer; transition: all 0.3s ease !important; }
.info-card:hover { transform: translateY(-10px) !important; border: 2px solid #003399 !important; box-shadow: 0 25px 50px rgba(0,0,0,0.15) !important; }
.info-card-icon { width: 50px; height: 50px; background: #eff6ff; color: #003399; border-radius: 12px; display: flex; justify-content: center; align-items: center; font-size: 24px; }
.info-card-text h5 { margin: 0 0 5px 0; color: #0f172a; font-size: 16px; font-weight: 700; }
.info-card-text p { margin: 0; font-size: 13px; color: #64748b; }

/* Định dạng Container chính và các Panel tính năng */
.blue-panel {
    background-color: #3b82f6 !important; 
    padding: 30px !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15) !important;
    
    --background-fill-primary: transparent !important;
    --background-fill-secondary: transparent !important;
    --block-background-fill: transparent !important;
    --panel-background-fill: transparent !important;
}

.blue-panel > div, 
.blue-panel > div > div,
.blue-panel .gr-form,
.blue-panel .gr-box,
.blue-panel .gr-block {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.blue-panel label span, 
.blue-panel .gr-input-label,
.blue-panel .gr-checkbox-label {
    color: #ffffff !important;
    font-weight: 700 !important;
    background-color: transparent !important;
}

.blue-panel input[type="text"], 
.blue-panel input[type="number"], 
.blue-panel textarea, 
.blue-panel select,
.blue-panel .gr-dropdown > div {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    color: #0f172a !important;
    border-radius: 8px !important;
}

.blue-panel .options, 
.blue-panel .options li {
    background-color: #ffffff !important;
    color: #0f172a !important;
}

.blue-panel input[type="checkbox"] {
    -webkit-appearance: checkbox !important;
    -moz-appearance: checkbox !important;
    appearance: checkbox !important;
    width: 18px !important;
    height: 18px !important;
    cursor: pointer !important;
    display: inline-block !important;
    visibility: visible !important;
    opacity: 1 !important;
    margin-right: 8px !important;
    background-color: #ffffff !important;
}

.blue-panel .gr-checkbox {
    pointer-events: auto !important;
}

.blue-panel .progress-text,
.blue-panel .meta-text,
.blue-panel [class*="loading"] {
    color: #0f172a !important;
}

/* Định dạng cấu trúc và hiệu ứng của khối Footer */
.medvill-footer { 
    background: linear-gradient(rgba(0, 51, 153, 0.85), rgba(0, 20, 90, 0.95)), url('https://images.unsplash.com/photo-1519494026892-80bbd2d6fd0d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80') center/cover !important; 
    padding: 60px 8% !important; 
    display: flex !important; 
    justify-content: space-between !important; 
    align-items: center !important;
    color: white !important; 
    box-sizing: border-box !important;

    width: 100vw !important; 
    position: relative !important; 
    left: 50% !important; 
    right: 50% !important; 
    margin-left: -50vw !important; 
    margin-right: -50vw !important; 
    
    margin-bottom: -50px !important; 
}

.stat-box { 
    display: flex !important; 
    align-items: center !important; 
    justify-content: center !important; 
    gap: 20px !important; 
    flex: 1 !important; 
    border-right: 2px solid rgba(255, 255, 255, 0.2) !important; 
    padding: 20px 0 !important;
}

.stat-box:last-child {
    border-right: none !important;
}

.stat-icon { font-size: 48px !important; color: #ffffff !important; }

.stat-text h3 { font-size: 46px !important; margin: 0 !important; font-weight: 800 !important; line-height: 1 !important; color: #ffffff !important; }

.stat-text p { margin: 8px 0 0 !important; color: #93c5fd !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 1px !important;}

/* Điều chỉnh màu nền và loại bỏ Dark Mode của Gradio */
.analyst-container { 
    padding: 60px 4% !important; 
    margin: 20px auto 80px auto !important; /* Tạo khoảng cách với footer */
    max-width: 1650px !important; /* Kéo rộng ra */
    width: 92% !important; 
    background-color: #ffffff !important; 
    border-radius: 30px !important; /* Bo tròn hơn chút */
    box-shadow: 0 25px 50px rgba(0, 51, 153, 0.1) !important; /* Đổ bóng chuyên nghiệp */
    position: relative !important;
    z-index: 20 !important;
}

/* Đồng bộ biến màu hệ thống sang tông xám đậm hơn */
:root, .dark, .dark .gradio-container {
    --background-fill-primary: #e5e7eb !important; /* Khớp với nền xám bên trên */
    --background-fill-secondary: #ffffff !important;
    --border-color-primary: #9ca3af !important; /* Viền đậm hơn một chút cho khớp */
    --block-background-fill: transparent !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #cbd5e1 !important;
    --body-text-color: #0f172a !important;
    --block-label-text-color: #0f172a !important;
    --panel-background-fill: transparent !important;
}

.app-title-container { text-align: center; margin: 0 auto 40px auto; max-width: 700px; }
.app-section-title { color: #0f172a !important; font-size: 38px; font-weight: 800; margin: 0 0 15px 0; text-transform: uppercase; text-align: center; }
.app-section-subtitle { color: #64748b !important; font-size: 16px; text-align: center; margin-bottom: 50px; line-height: 1.6; }
"""

# =====================================================================
# PHẦN 3: LẮP RÁP GIAO DIỆN GRADIO VÀ RÀNG BUỘC SỰ KIỆN
# =====================================================================

with gr.Blocks(css=custom_css, title="Healthcare AI Agent") as demo:
    # Render các thành phần Header và Hero Section
    gr.HTML(HEADER_HTML)
    gr.HTML(HERO_HTML)
    gr.HTML(INFO_CARDS_HTML)

    # Render khối chức năng chính của ứng dụng
    with gr.Column(elem_classes="analyst-container"):
        
        gr.HTML(MAIN_APP_TITLE_HTML)

        with gr.Row():
            # Cột nhập liệu biểu mẫu bệnh nhân
            with gr.Column(elem_classes="blue-panel"):
                name = gr.Textbox(label="Full Name")
                email = gr.Textbox(label="Email")
                age = gr.Number(label="Age", value=35)
                sex = gr.Dropdown(
                    choices=["male", "female"],
                    value="male",
                    label="Sex",
                )
                height_cm = gr.Number(label="Height (cm)", value=170)
                weight_kg = gr.Number(label="Weight (kg)", value=70)
                children = gr.Number(label="Children", value=1)
                smoker = gr.Dropdown(
                    choices=["yes", "no"],
                    value="no",
                    label="Smoker",
                )
                region = gr.Dropdown(
                    choices=["northeast", "northwest", "southeast", "southwest"],
                    value="southeast",
                    label="Region",
                )
                city = gr.Dropdown(
                    choices=["Seoul", "Hanoi", "Ho Chi Minh City", "Da Nang"],
                    value="Seoul",
                    label="City",
                )
                symptoms = gr.Textbox(
                    label="Describe Symptoms",
                    lines=4,
                    placeholder="Example: chest pain and sweating",
                )
                use_ai = gr.Checkbox(
                    label="Use local AI explanation (experimental)",
                    value=False,
                )

                analyze_button = gr.Button("Analyze")
                history_button = gr.Button("Show patient history")

            # Cột hiển thị kết quả phân tích
            with gr.Column(elem_classes="blue-panel"):
                detailed_analysis = gr.Textbox(
                    label="Full analysis",
                    lines=12,
                )
                insurance_box = gr.Textbox(
                    label="Insurance estimate",
                    lines=2,
                )
                summary_box = gr.Textbox(
                    label="Health insight summary",
                    lines=4,
                )
                ai_box = gr.Textbox(
                    label="Explanation layer output",
                    lines=8,
                )
                history_box = gr.Textbox(
                    label="Patient history",
                    lines=10,
                )
                doctors_box = gr.Textbox(
                    label="Doctor recommendations",
                    lines=6,
                )
                doctor_map = gr.HTML(label="Doctor map")

        # Thiết lập các Trigger Event cho nút bấm
        analyze_button.click(
            analyze,
            inputs=[name, email, age, sex, height_cm, weight_kg, children, smoker, region, symptoms, city, use_ai],
            outputs=[
                detailed_analysis,
                insurance_box,
                summary_box,
                ai_box,
                history_box,
                doctors_box,
                doctor_map,
            ],
        )

        history_button.click(
            show_history,
            inputs=[name, email],
            outputs=history_box,
        )

    # Render thành phần Footer
    gr.HTML(FOOTER_HTML)

if __name__ == "__main__":
    demo.launch(share=False, debug=True)