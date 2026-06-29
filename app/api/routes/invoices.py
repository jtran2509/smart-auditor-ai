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
import time


from app.models.invoice_processor import get_invoice_processor
from app.services.rag_service import get_rag_service
from app.agents.mcp_server import get_mcp_server
from app.services.llm_service import generate_audit_report
from app.services.ab_testing import get_ab_testing_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/invoices", tags=['Invoices'])

# Temporary storage for uploaded files
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# In-memory storage for processed results (in production,, use database)
processed_results: Dict[str, Dict[str, Any]] = {}

@router.post("/upload")
async def upload_invoice(file: UploadFile = File(...), background_tasks: BackgroundTasks = None
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
        mcp_server = get_mcp_server()
        vendor_risk = mcp_server.get_vendor_risk(extracted_data.get("company_name", ''))

        fraud_flags = []

        if vendor_risk and vendor_risk.get("risk_level") == "HIGH":
            fraud_flags.append({
                "type": "DUPLICATE",
                "severity": "HIGH",
                "details": f"Vendor {extracted_data.get('company_name')} has high risk score: {vendor_risk.get('risk_score')}"
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

        # Get a/b testing service
        ab_testing = get_ab_testing_service()

        # Determine which model to use (using filename as user_id)
        model_version = ab_testing.get_model_version(user_id=file.filename)

        start_time = time.time()

        if model_version == "model_a":
            # Use Regex
            processor = get_invoice_processor()
            extracted_data = processor.extract_fields_regex(file_path)
            success = len(extracted_data) > 0
        else:
            # Use Fine-tuned model
            processor = get_invoice_processor()
            extracted_data = processor.extract_with_model(file_path)
            success = len(extracted_data) > 0

            if not success:
                # Fall back to regex if model fails
                extracted_data = processor.extract_fields_regex(file_path)
                success = len(extracted_data) > 0
            
        processing_time = time.time() - start_time
    
            # log result
        ab_testing.log_result(model_version, success, processing_time, extracted_data)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "invoice_id": invoice_id,
                "extracted_fields": extracted_data,
                "fraud_flags": fraud_flags,
                "fraud_risk": "HIGH" if fraud_flags else "LOW",
                "ab_test_model": model_version,
                "processing_time_ms": round(processing_time * 1000, 2),
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

    
@router.post("/{invoice_id}/analyze")   
async def analyze_invoice(invoice_id: str):
    """
    Generate AI audit report for an invoice using Gemini LLM
    """
    if invoice_id not in processed_results:
        raise HTTPException(404, "Invoice not found")
    
    result = processed_results[invoice_id]
    # Import LLM Service
    from app.services.llm_service import generate_audit_report
    # Get vendor risk from MCP
    mcp_server = get_mcp_server()
    vendor_risk = mcp_server.get_vendor_risk(
        result['extracted_fields'].get('company_name', '')
    )

    # Generate report
    report = generate_audit_report(
        result['extracted_fields'],
        result.get('fraud_flags', []),
        vendor_risk=vendor_risk
    )
    
    return JSONResponse(status_code=200,
                        content={"success": True, 
                                 "invoice_id": invoice_id,
                                 "report": report})

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

@router.get("/ab-test/stats")
async def get_ab_test_stats():
    ab_testing = get_ab_testing_service()
    stats = ab_testing.get_statistics()
    return JSONResponse(content=stats)