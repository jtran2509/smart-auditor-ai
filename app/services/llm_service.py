import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variable
load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY not found. Please set it in .env file!")

def generate_audit_report(extracted_data, fraud_flags, vendor_risk):
    """
    Generate human-readable audit report using LLM
    """
    if not GEMINI_API_KEY:
        return " ⚠ AI report unavailable: Missing API key. Please get GEMINI_API_KEY is .env file."
    
    # prepare prompt
    prompt = f"""
    You are an AI auditor. Write a professional audit report based on:

    EXTRACTED INVOICE DATA:
    - Company: {extracted_data.get('company_name', 'N/A')}
    - Date: {extracted_data.get('date', 'N/A')}
    - Total Amount: {extracted_data.get('total_amount', 'N/A')}
    - Invoice Number: {extracted_data.get('invoice_number', 'N/A')}

    FRAUD CHECKS:
    - Duplicate detected: {len([f for f in fraud_flags if f['type'] == 'DUPLICATE']) > 0}
    - Vendor risk level: {vendor_risk.get('risk_level', 'LOW') if vendor_risk else "LOW"}

    Please write:
    1. Executive summary
    2. Recommendation
    3. Risk factors
    4. Next steps

    Keep it concise (max 100 words)
    """
    try:  
        model = genai.GenerativeModel('gemini-2.5-flash')
        response= model.generate_content(prompt)

        return response.text
    except Exception as e:
        return f" ❌ Error generating report: {str(e)}\n Please check your API key and internet connection."
