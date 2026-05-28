"""
Gradio UI for Smart Auditor AI
Provides web interface for invoice upload and fraud detection
"""
import gradio as gr
from fastapi import FastAPI
import requests
from pathlib import Path
import json
import os

# API endpoint
API_BASE_URL = "http://localhost:8000"

def process_invoice(file):
    """
    Process uploaded invoice file through fastAPI backend
    """
    if file is None:
        return " ❌ Please upload a file", "{}", "No file uploaded"
    try:
        with open(file.name, "rb") as f:
            files= {"file": (Path(file.name).name, f, "image/jpeg")}
            response = requests.post(f"{API_BASE_URL}/api/v1/invoices/upload", files=files)

        if response.status_code == 200:
            result= response.json()
        
            # Format extracted
            extracted = result.get('extracted_fields', {})
            formatted_fields = "\n".join([f"**{k.replace('_', ' ').title()}:** {v}" for k, v in extracted.items() if v])

            if not formatted_fields:
                formatted_fields = "⛔ No fields were extracted. The image might not be a valid invoice"

            # Format fraud flags
            fraud_flags = result.get("fraud_flags", [])
            fraud_risk = result.get("fraud_risk", "UNKNOWN")

            fraud_text = f"**Risk Level:** {fraud_risk}\n\n"
        
            if fraud_flags:
                fraud_text += "**⚠️ Fraud Alerts:**\n"
                for flag in fraud_flags:
                    fraud_text +=  f"- {flag.get('type', 'Unknown')}: {flag.get('details', '')}\n"
            else:
                fraud_text += "✅ No frauds indicators detected"

            # Format JSON for download
            json_output = json.dumps(result, indent=2, ensure_ascii=False)
            message = f"✅ **Invoice Processed Successfully!**\n\n**Invoice ID:** {result.get('invoice_id', 'N/A')}\n\n**Extracted Information:**\n{formatted_fields}"

            return message, fraud_text, json_output
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            return f"❌ **Error:** {error_msg}", "{}", f'{{"error": "{error_msg}"}}'
    except requests.exceptions.ConnectionError:
        return "❌ **Connection Error:** Make sure the FastAPI server is running on port 8000", "{}", '{"error": "Server not running"}'
    except Exception as e:
        return f"❌ **Unexpected Error:** {str(e)}", "{}", f'{{"error": "{str(e)}"}}'

def create_gradio_interface():
    """
    Create Gradio interface for Smart Auditor AI
    """
    with gr.Blocks(title="Smart Auditor AI") as demo:
        gr.Markdown("""
        # 📝 Smart Auditor AI
        ### Automated Invoice Fraud Detection System

        Upload an invoice image (JPG, PNG) and the system will:
        1. **Extract** key information (Company, Date, Amount, Invoice Number)
        2. **Detect** duplicates and fraud patterns using RAG
        3. **Generate** an audit report
        
        ---
        """)
        with gr.Row():
            with gr.Column(scale=1):
                file_input= gr.File(label="📂 Upload invoice", 
                                    file_types=['.pdf', '.png', '.jpg'],
                                    type="filepath")
                submit_btn = gr.Button(" 🔍 Analysing...", variant="primary")

                gr.Markdown("""
                ### 🗄 Example Invoices
                The system works best with clear, well-lit invoice images containing:
                - Company name
                - Date
                - Total amount
                - Invoice number
                """)

            with gr.Column(scale=2):
                output_text = gr.Markdown(label="📊 Analysis Results", value="Waiting for Upload...")
                fraud_output = gr.Markdown(label="🚨 Fraud Detection", value="No analysis performed yet.")

                with gr.Accordion("📝 Raw JSON Output", open=False):
                    json_output = gr.Textbox(label="API Reponse", lines=10, max_lines=20)

                # Add download button for jSON
                json_file = gr.File(label="Download report", visible=False)

        # Handle submission
        submit_btn.click(
            fn=process_invoice,
            inputs=file_input,
            outputs=[output_text, fraud_output, json_output]
            )
        # Status indicator
        gr.Markdown("""
        ---
        ### System status
        Make sure the FastAPI backend is running:
        ```bash
        uvicorn app.main:app --reload --port 8000      
        """)
        return demo

## For direct execution
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
