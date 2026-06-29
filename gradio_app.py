"""
Gradio UI for Smart Auditor AI
Provides web interface for invoice upload and fraud detection
"""
import gradio as gr
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import requests
from pathlib import Path
import json
import uuid
import shutil
import sys
import os
import base64
from io import BytesIO
from PIL import Image

invoice_id = None
sys.path.append(os.path.dirname(__file__))

# ======================
# FAST API 
# ======================
from app.models.invoice_processor import get_invoice_processor
from app.services.rag_service import get_rag_service

fastapi_app = FastAPI(title="Smart Auditor AI API", docs_url="/docs")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
processed_results = {}

# Sample images for dropdown (update paths to your actual images)
SAMPLE_IMAGES_DIR = Path("data/uploads")
SAMPLE_IMAGES = []

# Lấy 10 ảnh mẫu từ thư mục
if SAMPLE_IMAGES_DIR.exists():
    sample_files = sorted(SAMPLE_IMAGES_DIR.glob("*.jpg"))[:10]
    SAMPLE_IMAGES = [str(f) for f in sample_files]
else:
    # Fallback: nếu không tìm thấy thư mục
    SAMPLE_IMAGES = []


@fastapi_app.post("/api/v1/invoices/upload")
async def upload_invoice(file: UploadFile = File(...)):
    invoice_id = str(uuid.uuid4())[:8]
    file_path = UPLOAD_DIR / f"{invoice_id}.jpg"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    processor = get_invoice_processor()
    extracted_data = processor.extract_fields(file_path)

    rag_service = get_rag_service()
    duplicate_check = rag_service.check_duplicates(extracted_data)

    fraud_flags = []
    if duplicate_check:
        fraud_flags.append({
            "type": "DUPLICATE",
            "severity": "HIGH",
            "details": duplicate_check["reason"]
        })

    rag_service.add_invoice(extracted_data)

    result = {
        "invoice_id": invoice_id,
        "extracted_fields": extracted_data,
        "fraud_flags": fraud_flags,
        "fraud_risk": "HIGH" if fraud_flags else "LOW"
    }

    processed_results[invoice_id] = result
    
    return JSONResponse(content={
        "success": True, 
        **result
    })


@fastapi_app.post("/api/v1/invoices/upload-sample")
async def upload_sample_invoice(data: dict):
    """Upload a sample image from the dropdown list"""
    image_path = data.get("image_path")
    if not image_path:
        raise HTTPException(400, "No image path provided")
    
    file_path = Path(image_path)
    if not file_path.exists():
        raise HTTPException(404, f"Image not found: {image_path}")
    
    # Read image and upload
    with open(file_path, "rb") as f:
        # Create a mock UploadFile
        from fastapi import UploadFile
        mock_file = UploadFile(filename=file_path.name, file=f)
        return await upload_invoice(mock_file)


@fastapi_app.get("/api/v1/invoices/sample-images")
async def get_sample_images():
    """Return list of sample images for dropdown"""
    return JSONResponse(content={"images": SAMPLE_IMAGES})


@fastapi_app.get("/api/v1/invoices/{invoice_id}")
async def get_invoice(invoice_id: str):
    if invoice_id not in processed_results:
        raise HTTPException(404, "Invoice not found")
    return JSONResponse(content=processed_results[invoice_id])


@fastapi_app.get("/health")
async def health():
    return {"status": "ok"}


# =========================
# GRADIO UI (call FastAPI internal)
# =========================
@fastapi_app.post("/api/v1/invoices/{invoice_id}/analyze")   
async def analyze_invoice(invoice_id: str):
    """Generate AI audit report for an invoice using Gemini LLM"""
    if invoice_id not in processed_results:
        raise HTTPException(404, "Invoice not found")
    
    result = processed_results[invoice_id]
    # Import LLM Service
    from app.services.llm_service import generate_audit_report
    from app.agents.mcp_server import get_mcp_server
    
    mcp_server = get_mcp_server()
    vendor_risk = mcp_server.get_vendor_risk(
        result['extracted_fields'].get('company_name', '')
    )

    report = generate_audit_report(
        result['extracted_fields'],
        result.get('fraud_flags', []),
        vendor_risk=vendor_risk
    )
    
    return JSONResponse(status_code=200,
                        content={"success": True, 
                                 "invoice_id": invoice_id,
                                 "report": report})


API_BASE_URL = "http://localhost:7860" 

# Store current invoice_id for sample selection
current_sample_invoice_id = None


def load_sample_image(image_path: str):
    """Load and display sample image"""
    if not image_path:
        return None, "⚠️ Please select a sample image"
    
    try:
        # Return image path for display
        return image_path, f"📂 Selected: {Path(image_path).name}"
    except Exception as e:
        return None, f"❌ Error loading image: {str(e)}"


def process_sample_invoice(image_path: str):
    """Process sample invoice from dropdown"""
    global current_sample_invoice_id
    
    if not image_path:
        return "⚠️ Please select an image", "{}", "No file selected", None
    
    try:
        # Upload to FastAPI
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            response = requests.post(f"{API_BASE_URL}/api/v1/invoices/upload", files=files)
        
        if response.status_code == 200:
            result = response.json()
            current_sample_invoice_id = result.get("invoice_id", None)
            
            extracted = result.get('extracted_fields', {})
            # ====================== 
            formatted_fields = []
            for k, v in extracted.items():
                if v:
                    # Chuyển key từ snake_case sang Title Case
                    key_display = k.replace('_', ' ').title()
                    formatted_fields.append(f"**{key_display}:** {v}")
            
            if formatted_fields:
                # Nối các field bằng dấu xuống dòng
                formatted_text = "\n\n".join(formatted_fields)  # 2 dấu xuống dòng giữa các field
            else:
                formatted_text = "⚠️ No fields were extracted. The image might not be a valid invoice"
            # ========================

            fraud_flags = result.get("fraud_flags", [])
            fraud_risk = result.get("fraud_risk", "UNKNOWN")
            
            fraud_text = f"**Risk Level:** {fraud_risk}\n\n"
            if fraud_flags:
                fraud_text += "**⚠️ Fraud Alerts:**\n"
                for flag in fraud_flags:
                    fraud_text += f"- {flag.get('type', 'Unknown')}: {flag.get('details', '')}\n"
            else:
                fraud_text += "✅ No fraud indicators detected"
            
            json_output = json.dumps(result, indent=2, ensure_ascii=False)
            message = f"✅ **Invoice Processed Successfully!**\n\n**Invoice ID:** {result.get('invoice_id', 'N/A')}\n\n**Extracted Information:**\n\n{formatted_text}"
            
            return message, fraud_text, json_output, current_sample_invoice_id
        else:
            return f"❌ Error: {response.text}", "{}", "{}", None
            
    except Exception as e:
        return f"❌ Unexpected Error: {str(e)}", "{}", "{}", None


def process_invoice(file):
    """Process uploaded invoice file through fastAPI backend"""
    global invoice_id

    if file is None:
        return "❌ Please upload a file", "{}", "No file uploaded", None
    
    try:
        file_path = file.name if hasattr(file, "name") else str(file)

        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "image/jpeg")}
            response = requests.post(f"{API_BASE_URL}/api/v1/invoices/upload", files=files)

        if response.status_code == 200:
            result = response.json()
            invoice_id = result.get("invoice_id", None)
        
            extracted = result.get('extracted_fields', {})
            # ================

            formatted_fields = []
            for k, v in extracted.items():
                if v:
                    # Chuyển key từ snake_case sang Title Case
                    key_display = k.replace('_', ' ').title()
                    formatted_fields.append(f"**{key_display}:** {v}")
            
            if formatted_fields:
                # Nối các field bằng dấu xuống dòng
                formatted_text = "\n\n".join(formatted_fields)  # 2 dấu xuống dòng giữa các field
            else:
                formatted_text = "⚠️ No fields were extracted. The image might not be a valid invoice"

            # =============
            fraud_flags = result.get("fraud_flags", [])
            fraud_risk = result.get("fraud_risk", "UNKNOWN")

            fraud_text = f"**Risk Level:** {fraud_risk}\n\n"
        
            if fraud_flags:
                fraud_text += "**⚠️ Fraud Alerts:**\n"
                for flag in fraud_flags:
                    fraud_text += f"- {flag.get('type', 'Unknown')}: {flag.get('details', '')}\n"
            else:
                fraud_text += "✅ No fraud indicators detected"

            json_output = json.dumps(result, indent=2, ensure_ascii=False)
            message = message = f"✅ **Invoice Processed Successfully!**\n\n**Invoice ID:** {result.get('invoice_id', 'N/A')}\n\n**Extracted Information:**\n\n{formatted_text}"

            return message, fraud_text, json_output, invoice_id
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            return f"❌ **Error:** {error_msg}", "{}", f'{{"error": "{error_msg}"}}', None
    
    except Exception as e:
        return f"❌ **Unexpected Error:** {str(e)}", "{}", f'{{"error": "{str(e)}"}}', None


def generate_report(invoice_id):
    """Generate AI report for the processed invoice"""
    if not invoice_id:
        return "⚠️ Please upload and process an invoice first."
    try:
        response = requests.post(f"{API_BASE_URL}/api/v1/invoices/{invoice_id}/analyze")
        if response.status_code == 200:
            return response.json().get('report', 'No report generated')
        else:
            return f"❌ Error generating report: {response.json().get('detail', 'Unknown error')}"
    except Exception as e:
        return f"❌ Error: {str(e)}"


def create_gradio_interface():
    """Create Gradio interface for Smart Auditor AI"""
    with gr.Blocks(title="Smart Auditor AI", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧾 Smart Auditor AI
        ### Automated Invoice Fraud Detection System

        Upload an invoice image or select from sample receipts below.
        """)
        
        with gr.Tabs():
            
            with gr.TabItem("📋 Sample Receipts"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sample_dropdown = gr.Dropdown(
                            choices=SAMPLE_IMAGES,
                            label="📋 Select Sample Receipt",
                            info="Choose a sample receipt to test the system"
                        )
                        sample_image_preview = gr.Image(
                            label="📷 Receipt Preview",
                            type="filepath",
                            height=300
                        )
                        sample_status = gr.Markdown("💡 Select a receipt to analyze")
                    
                    with gr.Column(scale=2):
                        sample_output_text = gr.Markdown(
                            label="📊 Analysis Results", 
                            value="Select a sample receipt from the dropdown"
                        )
                        sample_fraud_output = gr.Markdown(
                            label="🚨 Fraud Detection", 
                            value="No analysis performed yet."
                        )

            with gr.TabItem("📤 Upload Invoice"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="📂 Upload invoice", 
                            file_types=['.jpg', '.jpeg', '.png'],
                            type="filepath"
                        )
                        upload_status = gr.Markdown("💡 Upload an image to analyze")
                    
                    with gr.Column(scale=2):
                        output_text = gr.Markdown(label="📊 Analysis Results", value="Waiting for upload...")
                        fraud_output = gr.Markdown(label="🚨 Fraud Detection", value="No analysis performed yet.")
            
        
        # Hidden store for invoice_id
        invoice_id_store = gr.Textbox(visible=False)
        
        # AI Report section (shared)
        with gr.Row():
            with gr.Column():
                report_btn = gr.Button("📄 Generate AI Audit Report", variant="secondary", size="lg")
                report_output = gr.Textbox(
                    label="🤖 AI Audit Report", 
                    lines=15, 
                    placeholder="Click 'Generate AI Audit Report' after analyzing an invoice..."
                )
        
        # ===== Event Handlers =====
        
        def auto_analyze_upload(file, progress=gr.Progress()):
            """Upload and analyze with progress bar"""
            if file is None:
                return "💡 Please upload an image", "{}", None
            
            progress(0.0, desc="Starting analysis...")
            
            # Step 1: Upload file
            progress(0.2, desc="Uploading file...")
            
            # call process invoice function 
            message, fraud_text, json_output, inv_id = process_invoice(file)
            
            progress(0.6, desc="Extracting information with LayoutLMv3...")
            progress(0.9, desc="Finalizing results...")
            
            # Nếu có lỗi, hiển thị thông báo
            if "Error" in message or "Connection" in message:
                progress(1.0, desc="Analysis failed")
                return message, fraud_text, None
            
            progress(1.0, desc="✅ Analysis complete!")
            return message, fraud_text, inv_id
        
        # Upload: auto analyze with progress
        file_input.change(
            fn=auto_analyze_upload,
            inputs=file_input,
            outputs=[output_text, fraud_output, invoice_id_store]
        )
        
        # Sample: auto analyze with progress
        def auto_analyze_sample(image_path, progress=gr.Progress()):
            if not image_path:
                return None, "💡 Please select a receipt", "No analysis performed yet.", None
            
            progress(0.0, desc="Loading image...")
            preview = image_path
            progress(0.3, desc="Analyzing with LayoutLMv3...")
            
            message, fraud_text, json_output, inv_id = process_sample_invoice(image_path)
            
            progress(0.7, desc="Extracting information...")
            progress(0.9, desc="Finalizing...")
            progress(1.0, desc="✅ Done!")
            
            return preview, message, fraud_text, inv_id
        
        sample_dropdown.change(
            fn=auto_analyze_sample,
            inputs=sample_dropdown,
            outputs=[sample_image_preview, sample_output_text, sample_fraud_output, invoice_id_store]
        )
        
        # Generate Report (shared)
        report_btn.click(
            fn=generate_report,
            inputs=invoice_id_store,
            outputs=report_output
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ About
        - **Supported formats**: Southeast Asian receipts (SROIE dataset)
        - **Note**: This is an MLOps demonstration project
        - **Tech stack**: LayoutLMv3, FastAPI, Gradio, ChromaDB, Gemini AI
        """)
    
    return demo

# =======================
# MAIN: Run Gradio + FastAPI
# =======================
if __name__ == "__main__":
    demo = create_gradio_interface()
    app = gr.mount_gradio_app(fastapi_app, demo, path="/")
    uvicorn.run(app, host="0.0.0.0", port=7860)

# """
# Gradio UI for Smart Auditor AI
# Provides web interface for invoice upload and fraud detection
# """
# import gradio as gr
# import uvicorn
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import JSONResponse
# import requests
# from pathlib import Path
# import json
# import uuid
# import shutil
# import threading
# import time
# import sys
# import os

# invoice_id = None
# sys.path.append(os.path.dirname(__file__))

# # ======================
# # FAST API 
# # ======================
# from app.models.invoice_processor import get_invoice_processor
# from app.services.rag_service import get_rag_service

# fastapi_app = FastAPI(title="Smart Auditor AI API", docs_url="/docs")

# UPLOAD_DIR = Path("data/uploads")
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# processed_results = {}

# @fastapi_app.post("/api/v1/invoices/upload")
# async def upload_invoice(file: UploadFile= File(...)):
#     invoice_id = str(uuid.uuid4())[:8]
#     file_path = UPLOAD_DIR / f"{invoice_id}.jpg"

#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     processor = get_invoice_processor()
#     extracted_data = processor.extract_fields(file_path)

#     rag_service = get_rag_service()
#     duplicate_check = rag_service.check_duplicates(extracted_data)

#     fraud_flags = []
#     if duplicate_check:
#         fraud_flags.append({
#             "type": "DUPLICATE",
#             "severity": "HIGH",
#             "details": duplicate_check["reason"]
#         })

#     rag_service.add_invoice(extracted_data)

#     result = {
#         "invoice_id": invoice_id,
#         "extracted_fields": extracted_data,
#         "fraud_flags": fraud_flags,
#         "fraud_risk": "HIGH" if fraud_flags else "LOW"
#     }

#     processed_results[invoice_id] = result
    
#     return JSONResponse(content={
#         "success": True, 
#         **result
#     })

# @fastapi_app.get("/api/v1/invoices/{invoice_id}")
# async def get_invoice(invoice_id: str):
#     if invoice_id not in processed_results:
#         raise HTTPException(404, "Invoice not found")
#     return JSONResponse(content=processed_results[invoice_id])

# @fastapi_app.get("/health")
# async def health():
#     return {"status": "ok"}

# # =========================
# # GRADIO UI (call FastAPI internal)
# # =========================
# @fastapi_app.post("/api/v1/invoices/{invoice_id}/analyze")   
# async def analyze_invoice(invoice_id: str):
#     """
#     Generate AI audit report for an invoice using Gemini LLM
#     """
#     if invoice_id not in processed_results:
#         raise HTTPException(404, "Invoice not found")
    
#     result = processed_results[invoice_id]
#     extracted = result.get("extracted_fields", {})
#     # Import LLM Service
#     from app.services.llm_service import generate_audit_report
#     from app.agents.mcp_server import get_mcp_server
    
#     # Get vendor risk from MCP
#     mcp_server = get_mcp_server()
#     vendor_risk = mcp_server.get_vendor_risk(
#         result['extracted_fields'].get('company_name', '')
#     )

#     # Generate report
#     report = generate_audit_report(
#         result['extracted_fields'],
#         result.get('fraud_flags', []),
#         vendor_risk=vendor_risk
#     )
    
#     return JSONResponse(status_code=200,
#                         content={"success": True, 
#                                  "invoice_id": invoice_id,
#                                  "report": report})


# API_BASE_URL = "http://localhost:7860" 

# def process_invoice(file):
#     """
#     Process uploaded invoice file through fastAPI backend
#     """
#     global invoice_id

#     if file is None:
#         return " ❌ Please upload a file", "{}", "No file uploaded", None
    
#     try:
#         file_path = file.name if hasattr(file, "name") else str(file)

#         with open(file_path, "rb") as f:
#             files= {"file": (Path(file_path).name, f, "image/jpeg")}
#             response = requests.post(f"{API_BASE_URL}/api/v1/invoices/upload", files=files)

#         if response.status_code == 200:
#             result= response.json()

#             # Save invoice_id for later use
#             invoice_id = result.get("invoice_id", None)
        
#             # Format extracted
#             extracted = result.get('extracted_fields', {})
#             formatted_fields = "\n".join([f"**{k.replace('_', ' ').title()}:** {v}" for k, v in extracted.items() if v])

#             if not formatted_fields:
#                 formatted_fields = "⛔ No fields were extracted. The image might not be a valid invoice"

#             # Format fraud flags
#             fraud_flags = result.get("fraud_flags", [])
#             fraud_risk = result.get("fraud_risk", "UNKNOWN")

#             fraud_text = f"**Risk Level:** {fraud_risk}\n\n"
        
#             if fraud_flags:
#                 fraud_text += "**⚠️ Fraud Alerts:**\n"
#                 for flag in fraud_flags:
#                     fraud_text +=  f"- {flag.get('type', 'Unknown')}: {flag.get('details', '')}\n"
#             else:
#                 fraud_text += "✅ No frauds indicators detected"

#             # Format JSON for download
#             json_output = json.dumps(result, indent=2, ensure_ascii=False)
#             message = f"✅ **Invoice Processed Successfully!**\n\n**Invoice ID:** {result.get('invoice_id', 'N/A')}\n\n**Extracted Information:**\n{formatted_fields}"

#             return message, fraud_text, json_output, invoice_id
#         else:
#             error_msg = response.json().get('detail', 'Unknown error')
#             return f"❌ **Error:** {error_msg}", "{}", f'{{"error": "{error_msg}"}}', None
    
#     except Exception as e:
#         return f"❌ **Unexpected Error:** {str(e)}", "{}", f'{{"error": "{str(e)}"}}', None
    
# def generate_report(invoice_id):
#     """
#     Generate AI report for the processed invoice
#     """
#     if not invoice_id:
#         return " ⚠ Please upload and process an invoice first."
#     try:
#         response= requests.post(f"{API_BASE_URL}/api/v1/invoices/{invoice_id}/analyze")

#         if response.status_code == 200:
#             return response.json().get('report', 'No report generated')
#         else:
#             return f" Error generating report: {response.json().get('detail', 'Unknown error')}"
#     except Exception as e:
#         return f" ❌ Error: {str(e)}"

# def show_ab_stats():
#     response = requests.get(f"{API_BASE_URL}/api/v1/invoices/ab-test/stats")
#     if response.status_code == 200:
#         stats = response.json()
#         text = "## A/B Testing Results\n\n"
#         text+= "| Model | Accuracy | Avg Time (ms) | Samples | \n"
#         text+= "| ------| -------- | ------------- | ------- |\n"

#         for key, data in stats.items():
#             if key in ['model_a', 'model_b']:
#                 text+= f"| {data['name']} | {data['accuracy']}% | {data['avg_time_ms']} | {data['total']} |\n"

#         if "uplift_percent" in stats:
#             text+= f"\n 🚀 Uplift: {stats['uplift_percent']}% improvement with fine-tuned model!"

#         return text
#     return "No data available"
        
# def create_gradio_interface():
#     """
#     Create Gradio interface for Smart Auditor AI
#     """
#     with gr.Blocks(title="Smart Auditor AI") as demo:
#         gr.Markdown("""
#         # 📝 Smart Auditor AI
#         ### Automated Invoice Fraud Detection System

#         Upload an invoice image (JPG, PNG) and the system will:
#         1. **Extract** key information (Company, Date, Amount, Invoice Number)
#         2. **Detect** duplicates and fraud patterns using RAG
#         3. **Generate** an audit report
        
#         ---
#         """)
#         with gr.Row():
#             with gr.Column(scale=1):
#                 file_input= gr.File(label="📂 Upload invoice", 
#                                     file_types=['.pdf', '.png', '.jpg'],
#                                     type="filepath")
#                 submit_btn = gr.Button(" 🔍 Analysing...", variant="primary")

#                 gr.Markdown("""
#                 ### 🗄 Example Invoices
#                 The system works best with clear, well-lit invoice images containing:
#                 - Company name
#                 - Date
#                 - Total amount
#                 - Invoice number
#                 """)

#             with gr.Column(scale=2):
#                 output_text = gr.Markdown(label="📊 Analysis Results", value="Waiting for Upload...")
#                 fraud_output = gr.Markdown(label="🚨 Fraud Detection", value="No analysis performed yet.")

#                 with gr.Accordion("📝 Raw JSON Output", open=False):
#                     json_output = gr.Textbox(label="API Reponse", lines=10, max_lines=20)

#         invoice_id_store = gr.Textbox(visible=False)
        
#         # AI report section
#         with gr.Row():
#             with gr.Column():
#                 # Adding buttong and new output
#                 report_btn = gr.Button("📝 Generate AI Audit report", variant="secondary", size="lg")
#                 report_output = gr.Textbox(label="🤖 AI Audit Report", lines=15, placeholder="Click 'Generate AI Audit Report' after analyzing an invoice...")

#         # with gr.Row():
#         #     with gr.Row():
#         #         ab_stats_btn = gr.Button("📊 Show A/B Test Statistics")
#         #         ab_stats_output = gr.Markdown()
                            
#         # Handle submission
#         submit_btn.click(
#             fn=process_invoice,
#             inputs=file_input,
#             outputs=[output_text, fraud_output, json_output, invoice_id_store]
#             )
        
#         report_btn.click(
#             fn=generate_report,
#             inputs=invoice_id_store,
#             outputs=report_output
#         )
#         # ab_stats_btn.click(fn=show_ab_stats, outputs=ab_stats_output)

#         # Status indicator
#         gr.Markdown("""
#         ---
#         ### System status
#         Make sure the FastAPI backend is running:
#         ```bash
#         uvicorn app.main:app --reload --port 8000      
#         """)
#     return demo

# # =======================
# # MAIN: Run Gradio + FastAPI
# # =======================
# if __name__ == "__main__":
#     # Create Gradio interface
#     demo = create_gradio_interface()
#     # Mount Gradio into FastAPI    
#     app = gr.mount_gradio_app(fastapi_app, demo, path="/")

#     # Run FastAPI (blocking)
#     uvicorn.run(app, host="0.0.0.0", port=7860)    
