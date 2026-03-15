import gradio as gr
from core.diagnostics import SymptomDiagnosticEngine
from core.insurance import InsuranceEstimator
from core.doctors import find_doctors, create_doctor_map
from core.memory import generate_patient_id, update_patient_memory

# Khởi tạo các bộ máy AI từ các module đã hoàn thành
diag_engine = SymptomDiagnosticEngine()
ins_estimator = InsuranceEstimator()

def process_request(name, email, age, sex, bmi, children, smoker, region, symptoms, city):
    # 1. Tạo ID bệnh nhân và lấy lịch sử khám
    patient_id = generate_patient_id(name, email)
    
    # 2. Chẩn đoán triệu chứng dựa trên AI Engine
    diag_result = diag_engine.diagnose(symptoms, patient_id)
    top_diag = diag_result["top_diagnosis"]
    
    # 3. Ước tính bảo hiểm dựa trên mô hình đã huấn luyện
    price, ins_msg = ins_estimator.estimate(patient_id, age, sex, bmi, children, smoker, region)
    
    # 4. Tìm bác sĩ và tạo bản đồ HTML
    specialty = top_diag["specialty"] if top_diag else "Internal Medicine"
    doctors = find_doctors(specialty, city)
    doc_map_html = create_doctor_map(doctors)
    
    # 5. Lưu kết quả vào bộ nhớ (Memory)
    update_patient_memory(patient_id, {"name": name, "age": age}, symptoms, diag_result, price, doctors)
    
    # Định dạng nội dung hiển thị cho giao diện
    analysis = f"### Diagnosis: {top_diag['diagnosis'] if top_diag else 'N/A'}\n"
    analysis += f"**Triage Level:** {top_diag['triage_level'].upper() if top_diag else 'N/A'}\n"
    analysis += f"**Advice:** {top_diag['advice'] if top_diag else 'N/A'}"
    
    return analysis, ins_msg, diag_result["history_summary"], doc_map_html

# Xây dựng giao diện QuanSkill's Healthcare Agent bằng Gradio
with gr.Blocks(title="QuanSkill's Healthcare Agent") as demo:
    gr.Markdown("# QuanSkill's Healthcare Agent")
    gr.Markdown("Prototype decision-support assistant for symptoms, triage, insurance estimates, and doctor lookup.")
    
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="Name")
            email = gr.Textbox(label="Email")
            age = gr.Number(label="Age", value=25)
            sex = gr.Radio(["male", "female"], label="Sex")
            bmi = gr.Number(label="BMI", value=22.0)
            children = gr.Number(label="Children", value=0)
            smoker = gr.Radio(["yes", "no"], label="Smoker", value="no")
            region = gr.Dropdown(["southwest", "southeast", "northwest", "northeast"], label="Region")
            symptoms = gr.Textbox(label="Describe symptoms", placeholder="e.g., chest pain, sweating")
            city = gr.Textbox(label="City for doctor search", value="Seoul")
            submit_btn = gr.Button("Run AI Analysis", variant="primary")
            
        with gr.Column():
            full_analysis = gr.Markdown(label="Full analysis")
            ins_estimate = gr.Textbox(label="Insurance estimate")
            diag_summary = gr.Textbox(label="Diagnosis summary (History)")
            map_display = gr.HTML(label="Doctor Map")

    submit_btn.click(
        process_request, 
        inputs=[name, email, age, sex, bmi, children, smoker, region, symptoms, city], 
        outputs=[full_analysis, ins_estimate, diag_summary, map_display]
    )

if __name__ == "__main__":
    demo.launch()
