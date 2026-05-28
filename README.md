# Smart Auditor AI - Invoice Fraud Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44+-orange.svg)](https://gradio.app/)
[![CI Tests](https://github.com/jtran2509/smart-auditor-ai/actions/workflows/test.yml/badge.svg)](https://github.com/jtran2509/smart-auditor-ai/actions/workflows/test.yml)
[![Deploy](https://github.com/jtran2509/smart-auditor-ai/actions/workflows/deploy.yml/badge.svg)](https://github.com/jtran2509/smart-auditor-ai/actions/workflows/deploy.yml)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

## 📌 Overview
Smart Auditor AI is an end-to-end invoice fraud detection system that automatically:
- **Extracts** key  information from invoice messages (Company, Date, Total Amount, Invoice number).
- **Detects** duplicates and fradulent patterns using RAG (Retrieval Augmented Generation)
- **Generates** audit reports with LLM-powered explanations

## 🏗️ System architecture

## ✨ Features

| Feature | Description | Status |
|---------|-------------|--------|
| Invoice Upload | Upload JPG/PNG/PDF via API or UI | ✅ |
| OCR Extraction | Tesseract + LayoutLMv3 | ✅ |
| Fine-tuned Model | LayoutLMv3 on SROIE dataset (331 samples) | ✅ |
| RAG Storage | ChromaDB vector database | ✅ |
| Fraud Detection | Duplicate & similarity check | ✅ |
| MCP Server | Safe LLM database querying | ✅ |
| Gradio UI | Web interface for demo | ✅ |

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
Tesseract OCR (for Windows: download from UB-Mannheim)
```
### Installation
```bash
git clone https://github.com/yourusername/smart-auditor-ai.git
cd smart-auditor-ai
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Run the system
```bash
# Terminal 1: FastAPI Backend
uvicorn app.main:app --reload --port 8000
# Terminal 2: Gradio UI (optional, can also access via /demo)
python app/gradio_app.py
```
### Access
- API Docs: http://localhost:8000/docs
- Gradio UI: http://localhost:8000/demo (mounted)

## Model Performance
After fine-tuning on 264 SROIE sample
| Metric | Value |
|--------|-------|
| Train Samples | 264 |
| Test Samples | 67 |
| Validation Loss | 0.27 |
| F1 score | ~0.65 |

## 🛠 Tech stack
| Component | Technology |
|---------|-------------|
| Backend API | Fast API |
| Document AI | LayoutLMv3 + Tesseract |
| Vector DB | Chroma DB |
| RAG Framework | Langchain |
| UI | Gradio |
| ML Framework | PyTorch + Transformers |

## 📂 Project Structure
smart-auditor-ai/
├── app/
│   ├── api/           # FastAPI endpoints
│   ├── models/        # LayoutLMv3 processor
│   ├── services/      # RAG, MCP services
│   └── gradio_app.py  # Web UI
├── data/              # Training data (gitignored)
├── models/            # Fine-tuned model
├── scripts/           # Data preparation
├── tests/             # Unit tests
└── README.md

## License
MIT