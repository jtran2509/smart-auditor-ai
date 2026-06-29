"""
Test RAG service with sample queries
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.rag_service import get_rag_service

def test_rag():
    print('=' * 20)
    print("Testing RAG Service...")
    print("="*20)

    rag= get_rag_service()

    # Test query
    test_invoice = {
        "company_name": "UNIHAKKA INTERNATIONAL SDN BHD",
        "total_amount": "1250.00",
        "invoice_number": "INV-2024-001"
    }
    print(f"\n 🔍Searching for similar invoices...")
    print(f"    Company: {test_invoice['company_name']}")

    similar = rag.find_similar_invoices(test_invoice, n_results=3)

    if similar:
        print(f"\n Found {len(similar)} similar invoices.")
        for i, inv in enumerate(similar, 1):
            print(f"\n {i}. Similarity: {inv['similarity_score']:.2f}")
            print(f"    Company: {inv['metadata'].get('company_name', 'N/A')}")
            print(f"    Amount: {inv['metadata'].get('total_amount', 'N/A')}")
    else:
        print("\n ❌ No similar invoices found!")
    
    # test duplicate check
    print("\n" + "=" * 20)
    print("Testing duplicate detections...")

    duplicate_test = {
        "company_name": "UNIHAKKA INTERNATIONAL SDN BHD",
        "total_amount": "1250.00",
        "invoice_number": "INV-2024-001"  # Same as seeded data
    }
    result = rag.check_duplicates(duplicate_test)
    if result:
        print(f" Duplicate detected: {result['reason']}")
    else:
        print(f" No duplicate detected")

if __name__ == "__main__":
    test_rag()