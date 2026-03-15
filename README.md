# ThermaLVM: Edge-Deployable Industrial Energy Audit

[![CI](https://github.com/amiraghabiglou/ThermaLVM/actions/workflows/ci.yml/badge.svg)](https://github.com/amiraghabiglou/ThermaLVM/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An edge-deployable, multi-modal AI system designed to analyze synchronized high-resolution thermal and RGB imagery of industrial sites. It detects energy loss anomalies, cross-references findings with structural building codes via Retrieval-Augmented Generation (RAG), and executes deterministic compliance evaluations.

## 🏗️ Architecture (Phase 1: Zero-Shot Edge Inference)

This pipeline strictly decouples the heavy Vision-Language Model (VLM) inference from the application logic to ensure edge-deployability and scalable serving.

1. **Reasoning Engine (VLM):** A 4-bit quantized Qwen2-VL-7B model served via `llama.cpp` using Metal/CUDA acceleration. Generates strict, constrained JSON outputs (bounding boxes, defect types).
2. **Knowledge Base (RAG):** An embedded LanceDB vector store containing regional building codes and energy standards.
3. **API Gateway (FastAPI):** Orchestrates the data flow, maps JSON model outputs to Pydantic schemas, and executes deterministic PASS/FAIL mathematical compliance checks.
4. **Frontend:** Gradio interface for dual-image rendering and bounding box visualization.

## 📁 Repository Structure: ThermaLVM
```text
.
├── .github/workflows/
│   └── ci.yml                     # Future-proofing: CI/CD for testing the API
├── data/
│   ├── sample_images/             # RGB and Thermal pairs for testing
│   └── regulations/               # PDFs/Text of energy standards for RAG
├── docker/
│   ├── Dockerfile.frontend
│   ├── Dockerfile.api
│   └── Dockerfile.vllm            # Environment for Qwen2-VL/llama.cpp
├── src/
│   ├── api/                       # FastAPI gateway (handles requests, orchestrates RAG)
│   ├── frontend/                  # Gradio/Streamlit UI for dual-image upload
│   └── rag/                       # Vector DB ingestion and retrieval logic
├── .gitignore
├── docker-compose.yml             # Local deployment orchestration
├── requirements.txt
└── README.md
```
## 📊 Domain Standards & Validation
This system is engineered to evaluate physical building states against strict structural standards:
* **UK Building Regulations Part L:** Conservation of fuel and power.
* **ASHRAE 90.1:** Energy Standard for Buildings Except Low-Rise Residential Buildings.

**Recommended Validation Datasets:**
* **PST900 RGB-T Dataset:** For confined, subterranean, or indoor industrial environments.
* **Caltech Aerial RGB-Thermal (CART):** For exterior building envelope analysis via UAV.

## 🚀 Local Deployment (Apple Silicon / Edge Simulator)

You cannot simply boot the containers; the inference engine requires local model weights, and the RAG system requires an initialized vector database. Follow this exact sequence.

### 1. Download Model Weights
This system requires a quantized language model and a high-fidelity vision projector. Use the Hugging Face CLI to download the required GGUF artifacts directly into the `models/` directory.

```bash
pip install -U "huggingface_hub[cli]"

# Download the 4-bit quantized Qwen2-VL-7B reasoning engine
hf download bartowski/Qwen2-VL-7B-Instruct-GGUF Qwen2-VL-7B-Instruct-Q4_K_M.gguf --local-dir models/

# Download the fp16 multimodal vision projector
hf download bartowski/Qwen2-VL-7B-Instruct-GGUF mmproj-Qwen2-VL-7B-Instruct-f16.gguf --local-dir models/
```
### 2. Initialize the Knowledge Base (RAG)
Place your regulatory PDFs (e.g., UK Part L) into data/regulations/. Then, build the LanceDB vector store locally.

```bash
# Install local ingestion dependencies
pip install -r requirements.rag.txt

# Run the ingestion pipeline
python src/rag/ingest.py
```
### 3. Boot the Infrastructure
Once the .gguf weights are present and the .lance database is built, launch the decoupled microservices.

```bash
docker-compose up -d --build
```


### 4. Services Available
- Web Interface: http://localhost:8501 (Primary Entry Point)

- API Gateway: http://localhost:8000/docs (Swagger UI)

- VLM Inference Server: http://localhost:8080/v1 (OpenAI-Compatible Endpoint)