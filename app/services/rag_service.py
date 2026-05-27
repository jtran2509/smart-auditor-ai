"""
RAG Service for storing and retrieving invoice history
Uses ChromaDB for vector similarity search
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)

class RAGService:
    """
    Service for invoice history storage and similarity search
    """
    def __init__(self, persist_dir: str = 'data/chromadb'):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection for invoices
        self.collection_name = "invoice_history"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

        logger.info(f"RAGService initialized with {self.collection.count()} existing records")

    def generate_invoice_hash(self, invoice_data: Dict) -> str:
        """
        Generate unique hash for invoice to detect duplicates
        """
        # Create unique string from key fields
        unique_strings = f"{invoice_data.get('company_name', '')}_{invoice_data.get('invoice_number', '')}_{invoice_data.get('total_amount', '')}"
        return hashlib.md5(unique_strings.encode()).hexdigest()
    
    def add_invoice(self, invoice_data: Dict) -> str:
        """
        Add an invoice to the vector store

        Args:
            invoice_data: Dictionary with extracted fields
        Returns:
            Invoice with hash ID
        """
        try:
            # Generate hash ID
            invoice_hash = self.generate_invoice_hash(invoice_data)

            # Prepare text for embeddings (combine all fields)
            document_text = f"""
            Company: {invoice_data.get('company_name', '')}
            Date: {invoice_data.get('date', '')}
            Total Amount: {invoice_data.get('total_amount', '')}
            Invoice Number: {invoice_data.get('invoice_number', '')}
            Address: {invoice_data.get('address', '')}
            """

            # Add metadata
            metadata = {
                "company_name": invoice_data.get('company_name', ''),
                "invoice_number": invoice_data.get('invoice_number', ''),
                "total_amount": invoice_data.get('total_amount', ''),
                "date": invoice_data.get('date', ''),
                "timestamp": datetime.now().isoformat()
            }

            # Add to collection
            self.collection.upsert(
                ids=[invoice_hash],
                documents=[document_text],
                metadatas=[metadata]
            )

            logger.info(f"Added invoice {invoice_hash} to RAD store")
            return invoice_hash
        except Exception as e:
            logger.error(f"Error adding invoice to RAG:{str(e)}")
            return ""
        
    def find_similar_invoices(self, invoice_data: Dict, n_results: int = 5) -> List[Dict]:
        """
        Find similar invoices in history
        Args:
            invoice_data: Current invoice data
            n_results: Number of similar invoices to return
        Returns:
            Lists of similar invoices with similarity scores
        """
        try:
            # create query text
            query_text = f"""
            Company: {invoice_data.get('company_name', '')}
            Total Amount: {invoice_data.get('total_amount', '')}
            Invoice Number: {invoice_data.get('invoice_number', '')}
            """

            # Search for similar invoices
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )

            similar_invoices = []
            if results['ids'] and results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    similar_invoices.append({
                        "id": doc_id,
                        "similarity_score": float(1 - results['distances'][0][i]) if results['distances'] else 1.0,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "document": results['documents'][0][i] if results['documents'] else ""
                    })
            return similar_invoices
        except Exception as e:
            logger.error(f"Error finding similar invoices: {str(e)}")
            return []
        
    def check_duplicates(self, invoice_data: Dict) -> Optional[Dict]:
        """
        Check if invoice is duplicate (same invoice number or very similar)
        Returns:
            Duplicate invoice info if found, None otherwise
        """
        invoice_hash = self.generate_invoice_hash(invoice_data)

        # Check if exact hash exists
        existing = self.collection.get(ids=[invoice_hash])
        if existing['ids'] and len(existing['ids']) > 0:
            return {
                "is_duplicate": True,
                "reason": "Exact duplicate found",
                "existing_invoice": existing['metadatas'][0] if existing['metadatas'] else {}
            }
        
        # Check for similar invoices
        similar = self.find_similar_invoices(invoice_data, n_results=3)

        for sim in similar:
            # If similarity score > 0.9, flag as potential duplicate
            if sim['similarity_score'] > 0.9:
                return {
                    "is_duplicate": True,
                    "reason": f"Similar invoice found (similarity: {sim['similarity_score']:.2f})",
                    "similar_invoice": sim['metadata']
                }
            
        return None
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the RAG store
        """
        return {
            "total_invoices": self.collection.count(),
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_dir)
        }

# Singleton instance
_rag_service = None

def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service