"""
Model Context Protocol (MCP) Server
Provides safe database querying tools for LLM agents
"""
from typing import Dict, List, Any
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MCPServer:
    """
    MCP server that exposes safe tools for LLM agents to query data
    """
    def __init__(self, rag_service=None):
        self.rag_service = rag_service
        self.tools = {
            "get_invoice_history": self.get_invoice_history,
            "check_budget": self.check_budget,
            "get_vendor_risk": self.get_vendor_risk,
            "calculate_total_spent": self.calculate_total_spent
        }
        logger.info(f"MCP Server initialized with {len(self.tools)} tools")

    def get_tools_schema(self) -> List[Dict]:
        """
        Return Schema of available tools for LLM
        """
        return [
            {
               "name": "get_invoice_history",
                "description": "Get historical invoices for a specific company",
                "parameters": {
                    "company_name": "string (required) - Company name to search",
                    "limit": "integer (optional) - Number of results, default 10" 
                    }
            },
            {
                "name": "check_budget",
                "description": "Check remaining budget for a company/department",
                "parameters": {
                    "company_name": "string (required) - Company name",
                    "year": "integer (optional) - Year to check, default current year"
                }
            },
            {
                "name": "get_vendor_risk",
                "description": "Get risk score for a vendor based on history",
                "parameters": {
                    "vendor_name": "string (required) - Vendor name"
                }
            },
            {
                "name": "calculate_total_spent",
                "description": "Calculate total amount spent with a vendor",
                "parameters": {
                    "vendor_name": "string (required) - Vendor name",
                    "start_date": "string (optional) - Start date (YYYY-MM-DD)",
                    "end_date": "string (optional) - End date (YYYY-MM-DD)"
                }
            }
        ]
    def get_invoice_history(self, company_name: str, limit: int = 10) -> Dict:
        """
        Get invoice history for a company
        """
        try:
            if not self.rag_service:
                return {"error": "RAG Service not available"}
            
            # Query similar invoices
            test_invoices = {"company_name": company_name, "total_amount": "", "invoice_number": ""}
            similar = self.rag_service.find_similar_invoices(test_invoices, n_results= limit)

            return {
                "success": True,
                "company": company_name,
                "invoice_count": len(similar),
                "invoices": similar
            }
        except Exception as e:
            logger.error(f"Error getting invoices history: {str(e)}")
            return {"error": str(e)}
        
    def check_budget(self, company_name: str, year: int = None) -> Dict:
        """
        Mock budget check - in production, quqery actual database
        """
        if year is None:
            year = datetime.now().year

        # Mock budget data (in production, get from real DB)
        mock_budgets ={
            "UNIHAKKA INTERNATIONAL": {"budget": 50000, "spent": 12450, "remaining": 37550},
            "BOOK TALK": {"budget": 30000, "spent": 8900, "remaining": 21100},
            "DEFAULT": {"budget": 100000, "spent": 0, "remaining": 100000}
        }
        budget = mock_budgets.get(company_name, mock_budgets["DEFAULT"])

        return {
            "success": True,
            "company": company_name,
            "year": year,
            "total_budget": budget['budget'],
            "total_spent": budget['spent'],
            "remaining_budget": budget['remaining'],
            "usage_percentage": (budget["spent"] / budget["budget"]) * 100 if budget["budget"] > 0 else 0
        }
    def get_vendor_risk(self, vendor_name: str) -> Dict:
        """
        Get vendor risk score based on history
        """
        # Mock risk scores (in production)
        high_risk_vendors = ["UNIHAKKA INTERNATIONAL", "TEST VENDOR"]

        is_high_risk = vendor_name.upper() in [v.upper() for v in high_risk_vendors]

        return {
            "success": True,
            "vendor": vendor_name,
            "risk_score": 85 if is_high_risk else 25,
            "risk_level": "HIGH" if is_high_risk else "LOW",
            "risk_factors": [
                "Multiple invoices from same vendor" if is_high_risk else "No risk factors detected"
            ]
        }
    def calculate_total_spent(self, vendor_name: str, start_date: str = None, end_date: str = None) -> Dict:
        """
        Calculate total amount spend with a vendor
        """
        try:
            if not self.rag_service:
                return {"error": "RAG Service not available"}
            
            # Query all invoices for this vendor
            test_invoice = {"company_name": vendor_name, "total_amount": "", "invoice_number": ""}
            similar = self.rag_service.find_similar_invoices(test_invoice, n_results=100)

            total = 0
            for inv in similar:
                try:
                    amount = float(inv['metadata'].get('total_amount', 0))
                    total += amount
                except:
                    pass

            return {
                "success": True,
                "vendor": vendor_name,
                "total_spent": total,
                "invoice_count": len(similar),
                "average_amount": total / len(similar) if similar else 0
            }
        except Exception as e:
            logger.error(f"Error calculating total spent: {str(e)}")
            return {"error": str(e)}
        
    def execute_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """
        Execute a tool by name with given parameters
        """
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            return self.tools[tool_name](**parameters)
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return {"error": str(e)}
        
# Singleton
_mcp_server = None

def get_mcp_server():
    global _mcp_server
    if _mcp_server is None:
        from app.services.rag_service import get_rag_service
        _mcp_server = MCPServer(rag_service=get_rag_service())
    return _mcp_server