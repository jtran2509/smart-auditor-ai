"""
Invoice API Routes
Handles file upload, processing, and retrieval
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import uuid
from datetime import datetime
from typing import Dict, Any
import logging

from app.models.invoice_processor import get_invoice_processor
from app.services.rag_service import get_rag_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/invoices", tags=['Invoices'])

# Temporary storage for uploaded files
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage for processed results (in production,, use database)
processed_results: Dict[str, Dict[str, Any]] = {}

@router.post("/upload")
async def upload_invoice(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> JSONResponse:
    """
    Upload an image invoice (JPG/PNG/PDF) and extract information

    Args:
        file: The invoice file to process

    Returns:
        JSON with extraced fields and tracking ID
    """
    # Validate file type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".pdf"}
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique ID for this invoice
    invoice_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().isoformat()

    # Save upload file
    file_path = UPLOAD_DIR / f"{invoice_id}{file_extension}"

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved: {file_path}")

        # Process this invoice
        processor = get_invoice_processor()
        extracted_data = processor.extract_fields(file_path)
        # Check for fraud/duplicates
        rag_service = get_rag_service()

        # Check if this is a duplicate
        duplicate_check = rag_service.check_duplicates(extracted_data)

        fraud_flags = []

        if duplicate_check:
            fraud_flags.append({
                "type": "DUPLICATE",
                "severity": "HIGH",
                "details": duplicate_check["reason"]
            })

        rag_service.add_invoice(extracted_data)

        # Add metadata
        result = {
            "invoice_id": invoice_id,
            "filename": file.filename,
            "uploaded_at": timestamp,
            "file_path": str(file_path),
            "extracted_fields": extracted_data,
            "status": "completed"
        }

        # Store result
        processed_results[invoice_id] = result
        # Add fraud flags to results
        result['fraud_flags'] = fraud_flags
        result['fraud_risks'] = "HIGH" if fraud_flags else "LOW"

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "invoice_id": invoice_id,
                "extracted_fields": extracted_data,
                "fraud_flags": fraud_flags,
                "fraud_risk": "HIGH" if fraud_flags else "LOW",
                "message": "Invoice processed successfully"
            }
        )
    except Exception as e:
        logger.error(f"Error processing invoice: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing invoice: {str(e)}"
        )
    finally:
        await file.close()

@router.get("/{invoice_id}")
async def get_invoice_result(invoice_id: str) -> JSONResponse:
    """
    Get processed result for a specific invoice
    Args:
        invoice_id: The ID returned from upload endpoint

    Returns:
        JSON with extracted fields
    """
    if invoice_id not in processed_results:
        raise HTTPException(
            status_code=404,
            detail=f"Invoice {invoice_id} not found"
        )
    result = processed_results[invoice_id]

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "invoice_id": invoice_id,
            "extracted_fields": result["extracted_fields"],
            "uploaded_at": result["uploaded_at"],
            "filename": result["filename"]
        }
    )

@router.get("/")
async def list_invoices(limit: int= 10) -> JSONResponse:
    """
    List recent processed invoices
    """
    recent = list(processed_results.values())[-limit:]

    return JSONResponse(
        status_code=200,
        content={
           "success": True,
            "count": len(recent),
            "invoices": [
                {
                    "invoice_id": inv["invoice_id"],
                    "filename": inv["filename"],
                    "uploaded_at": inv["uploaded_at"],
                    "status": inv["status"]
                }
                for inv in recent
            ] 
        }  
    )
