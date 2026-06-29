"""
Seed ChromaDB with sample invoice data for testing
Run this code to populate the database
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.rag_service import get_rag_service

# Sample invoice data
sample_invoices = [
    {
        "company_name": "UNIHAKKA INTERNATIONAL SDN BHD",
        "date": "2024-01-15",
        "total_amount": "1250.00",
        "invoice_number": "INV-2024-001",
        "address": "12, Jalan Tampoi 7/4, Johor Bahru"
    },
    {
        "company_name": "UNIHAKKA INTERNATIONAL SDN BHD", 
        "date": "2024-02-20",
        "total_amount": "850.50",
        "invoice_number": "INV-2024-045",
        "address": "12, Jalan Tampoi 7/4, Johor Bahru"
    },
    {
        "company_name": "BOOK TALK SDN BHD",
        "date": "2024-01-10",
        "total_amount": "3500.00",
        "invoice_number": "BT-2024-089",
        "address": "53 & 55, Jalan Utama 34, Skudai, Johor"
    },
    {
        "company_name": "SANYU TRADING",
        "date": "2024-03-05",
        "total_amount": "2750.00",
        "invoice_number": "SY-2024-123",
        "address": "48, Jalan Industri, Kuala Lumpur"
    },
    {
       "company": "LIGHTROOM GALLERY SDN BHD",
        "date": "20/12/2017",
        "address": "NO: 28,JALAN ASTANA 1C, BANDAR BUKIT RAJA, 41050 KLANG SELANGOR D.E, MALAYSIA",
        "total": "73.00" 
    },
    {
        "company": "TEO HENG STATIONERY & BOOKS",
        "date": "23/01/2018",
        "address": "NO.53,JALAN BESAR,45600 BATANG BERJUNTAI SELANGOR DARUL EHSAN",
        "total": "18.00"
    },
    {
        "company": "MR. D.I.Y. (M) SDN BHD",
        "date": "14-03-18",
        "address": "LOT 1851-A & 1851-B, JALAN KPB 6, KAWASAN PERINDUSTRIAN BALAKONG, 43300 SERI KEMBANGAN, SELANGOR",
        "total": "37.10"
    },
    {
        "company": "99 SPEED MART S/B",
        "date": "24-01-18",
        "address": "LOT P.T. 2811, JALAN ANGSA, TAMAN BERKELEY 41150 KLANG, SELANGOR 1076-IJOK",
        "total": "2.50"
    },
    {
        "company": "C W KHOO HARDWARE SDN BHD",
        "date": "01-03-18",
        "address": "NO.50,JALAN PBS 14/11, KAWASAN PERINDUSTRIAN BUKIT SERDANG,",
        "total": "21.20"
    },
    {
        "company": "UNIHAKKA INTERNATIONAL SDN BHD",
        "date": "12 MAR 2018",
        "address": "12, JALAN TAMPOI 7/4,KAWASAN PERINDUSTRIAN TAMPOI,81200 JOHOR BAHRU,JOHOR",
        "total": "$8.20"
    },
    {
        "company": "S.H.H. MOTOR (SUNGAI RENGIT) SDN. BHD.",
        "date": "23-01-2019",
        "address": "NO. 343, JALAN KURAU, SUNGAI RENGIT, 81620 PENGERANG, JOHOR.",
        "total": "20.00"
    },
    {
        "company": "PERNIAGAAN ZHENG HUI",
        "date": "12/02/2018",
        "address": "NO.59 JALAN PERMAS 9/5 BANDAR BARU PERMAS JAYA 81760 JOHOR BAHRU",
        "total": "112.45"
    },
    {
        "company": "SAM SAM TRADING CO",
        "date": "29-12-2017",
        "address": "67,JLN MEWAH 25/63 TMN SRI MUDA, 40400 SHAH ALAM.",
        "total": "14.10"
    },
    {
        "company": "SYARIKAT PERNIAGAAN GIN KEE",
        "date": "02/12/2017",
        "address": "NO 290, JALAN AIR PANAS. SETAPAK, 53200, KUALA LUMPUR.",
        "total": "29.68"
    },
    {
        "company": "SYARIKAT PERNIAGAAN GIN KEE",
        "date": "11/01/2018",
        "address": "NO 290, JALAN AIR PANAS, SETAPAK, 53200, KUALA LUMPUR.",
        "total": "21.20"
    },
    {
        "company": "LITTLE CRAVINGS SDN BHD",
        "date": "18-03-2018",
        "address": "HQ: 7, JLN SS21/34, 47400 PJ",
        "total": "RM53.60"
    },
    {
        "company": "ELITETRAX MARKETING SDN BHD",
        "date": "11.02.18",
        "address": "LOT 1F-01&02,1ST FLR,PARADIGM MALL, NO. 1 JALAN SS 7/26A, KELANA JAYA, 47301 PETALING JAYA",
        "total": "60.00"
    },
    {
        "company": "ADVANCO COMPANY",
        "date": "17/01/2018",
        "address": "NO 1&3, JALAN WANGSA DELIMA 12, WANGSA LINK, WANGSA MAJU, 53300 KUALA LUMPUR",
        "total": "29.00"
    },
    {
        "company": "SINNATHAMBY HOLDINGS SDN. BHD.",
        "date": "06/02/2018",
        "address": "NO.17, 18 & 41, JALAN BESAR, 39100 BRINCHANG, CAMERON HIGHLANDS, PAHANG",
        "total": "13.50"
    },
    {
        "company": "POPULAR BOOK CO. (M) SDN BHD",
        "date": "05/03/18",
        "address": "NO 8, JALAN 7/118B, DESA TUN RAZAK 56000 KUALA LUMPUR, MALAYSIA",
        "total": "9.90"
    }
]

def seed_database():
    print("="*20)
    print("Seeding ChromaDB with sample invoice data...")
    print("="*20)

    rag_service = get_rag_service()

    for invoice in sample_invoices:
        rag_service.add_invoice(invoice)
        print(f" ✅ {invoice['company_name']} - {invoice['invoice_number']}")

    stats = rag_service.get_statistics()
    print(f"\n" + "=" * 20)
    print(f"📊 Database Statistics")
    print(f" Total invoices: {stats['total_invoices']}")
    print(f" Collection: {stats['collection_name']}")

if __name__ == "__main__":
    seed_database()