# ThermaLVM: Multi-Modal Industrial Energy Auditing

An edge-deployable, multi-modal AI system that analyzes synchronized high-resolution thermal and RGB imagery of industrial sites to detect energy loss, cross-reference building codes via RAG, and predict energy compliance bands.

## Architecture (Phase 1: Zero-Shot & Pipeline Engineering)

This pipeline strictly decouples the VLM inference from the application logic to ensure edge-deployability and scalable serving.

1. **Frontend:** Interactive UI for dual-image (Thermal/RGB) uploads.
2. **API Gateway (FastAPI):** Orchestrates the data flow and manages the RAG pipeline.
3. **Reasoning Engine (VLM):** A 4-bit quantized Vision-Language Model (Qwen2-VL) served via `llama.cpp`/`vLLM` to extract physical defects from imagery.
4. **Knowledge Base (RAG):** Local vector store containing regional building codes and energy standards.

## Local Deployment (M3 Pro / Apple Silicon)

This repository is configured to run locally using Docker and Metal-accelerated inference.

### 1. Boot the Infrastructure
```bash
docker-compose up -d --build
```

2. Services Available
Web Interface: http://localhost:8501

API Gateway: http://localhost:8000

VLM Inference Server: http://localhost:8080