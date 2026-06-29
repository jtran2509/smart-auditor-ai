"""
Main entry point for the Invoice Fraud Detection System
"""
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from gradio_app import create_gradio_interface
import pytesseract
from huggingface_hub import login

# Import routers
from app.api.routes import invoices
from gradio_app import create_gradio_interface

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Create FastAPI instance with metadata
app = FastAPI(
    title="Smart Auditor AI",
    description="Smart Auditor AI - Invoice Fraud Detection & Audit Automation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ======= MOUNT GRADIO ======
try:
    # Creata gradio interface
    demo = create_gradio_interface()
    # Mount gradio to FastAPI at /demo path
    app = gr.mount_gradio_app(app, demo, path="/demo")
    logging.info(" Gradio UI mounted at /demo")
except Exception as e:
    logging.error(f"❌ Failed to mount gradio: {e}")
# ===================


# Configure CORS
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["http://localhost:8501", "http://localhost:7860", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Include routers
app.include_router(invoices.router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "smart-auditor-ai"}

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Smart Auditor AI",
        "docs": "/docs",
        "demo": "/demo",
        "health": "/health",
        "endpoints": {
            "upload_invoice": "POST /api/v1/invoices/upload",
            "get_invoice": "GET /api/v1/invoices/{invoice_id}",
            "list_invoices": "GET /api/v1/invoices/"
        }
    }
