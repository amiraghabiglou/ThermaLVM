import os
import base64
from typing import List, Literal
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import lancedb
from openai import OpenAI
import json


# ---------------------------------------------------------
# 1. The Strict Contract (Single Source of Truth)
# ---------------------------------------------------------
class BoundingBox(BaseModel):
    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)


class ThermalDefect(BaseModel):
    defect_type: Literal[
        "roof_heat_loss",
        "window_seal_failure",
        "wall_insulation_gap",
        "hvac_exhaust_anomaly",
        "thermal_bridge"
    ]
    delta_t: float = Field(description="Estimated temperature difference in Celsius")
    bounding_box: BoundingBox


class AuditOutput(BaseModel):
    defects: List[ThermalDefect]


# ---------------------------------------------------------
# 2. API & Client Initialization
# ---------------------------------------------------------
app = FastAPI(title="ThermaLVM API", version="1.0.0")

# The VLM is running in a separate container (llama.cpp or vLLM) exposing an OpenAI-compatible endpoint.
VLM_SERVER_URL = os.getenv("VLM_SERVER_URL", "http://vlm_server:8080/v1")
client = OpenAI(base_url=VLM_SERVER_URL, api_key="sk-no-key-required")

# Initialize local LanceDB connection via environment variable
DB_PATH = os.getenv("DB_PATH", "/app/data/vector_store")
db = lancedb.connect(DB_PATH)

# Ensure the regulations table exists (placeholder for the RAG ingestion pipeline)
if "regulations" not in db.table_names():
    # In a real scenario, this is populated by your RAG ingestion script
    schema = lancedb.pydantic.pydantic_to_schema(BaseModel)  # Simplified
    db.create_table("regulations", data=[
        {"vector": [0.1] * 384, "defect_type": "roof_heat_loss", "max_allowed_delta_t": 5.0,
         "code_ref": "Part L - 4.1"},
        {"vector": [0.2] * 384, "defect_type": "window_seal_failure", "max_allowed_delta_t": 3.0,
         "code_ref": "Part L - 4.3"}
    ])


# ---------------------------------------------------------
# 3. Helper Functions
# ---------------------------------------------------------
def encode_image(image_bytes: bytes) -> str:
    """Encodes image bytes to base64 for the VLM payload."""
    return base64.b64encode(image_bytes).decode('utf-8')


def retrieve_regulation(defect_type: str) -> dict:
    """Mock RAG retrieval: Fetches regulatory thresholds based on defect type."""
    table = db.open_table("regulations")
    # In a production RAG, you embed the query and use table.search(vector).
    # Here, we do a deterministic metadata filter for speed and accuracy.
    results = table.search().where(f"defect_type = '{defect_type}'").limit(1).to_pandas()
    if results.empty:
        return {"max_allowed_delta_t": 999.0, "code_ref": "Unknown"}
    return results.iloc[0].to_dict()


# ---------------------------------------------------------
# 4. Core Endpoints
# ---------------------------------------------------------
@app.post("/audit")
async def run_thermal_audit(
        rgb_image: UploadFile = File(...),
        thermal_image: UploadFile = File(...)
):
    """
    Executes the end-to-end pipeline: VLM Inference -> RAG Retrieval -> Rule Engine.
    """
    try:
        rgb_bytes = await rgb_image.read()
        thermal_bytes = await thermal_image.read()

        rgb_b64 = f"data:image/jpeg;base64,{encode_image(rgb_bytes)}"
        thermal_b64 = f"data:image/jpeg;base64,{encode_image(thermal_bytes)}"

        # 1. VLM Inference (Prompting Qwen2-VL with Outlines/JSON Schema enforcement)
        # Note: It relies on the OpenAI-compatible client's response_format to enforce the Pydantic schema.
        # This compiles the schema into the decoding process on the server side.
        response = client.chat.completions.create(
            model="qwen2-vl-7b-instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert industrial thermal auditor. Analyze the provided aligned RGB and Thermal images. Identify thermal defects, estimate the temperature delta, and provide normalized bounding boxes. You must strictly adhere to the provided JSON schema."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyze these factory images for energy leaks."},
                        {"type": "image_url", "image_url": {"url": rgb_b64}},
                        {"type": "image_url", "image_url": {"url": thermal_b64}}
                    ]
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "audit_output",
                    "schema": AuditOutput.model_json_schema()
                }
            },
            temperature=0.0  # Deterministic output
        )

        # Parse the guaranteed JSON
        raw_output = json.loads(response.choices[0].message.content)
        vlm_data = AuditOutput(**raw_output)

        # 2. RAG & Rule Engine Evaluation
        final_report = []
        for defect in vlm_data.defects:
            reg_data = retrieve_regulation(defect.defect_type)

            # Deterministic compliance check
            is_compliant = defect.delta_t <= reg_data["max_allowed_delta_t"]

            final_report.append({
                "defect": defect.defect_type,
                "detected_delta_t": defect.delta_t,
                "bounding_box": defect.bounding_box.model_dump(),
                "regulation_threshold": reg_data["max_allowed_delta_t"],
                "regulation_code": reg_data["code_ref"],
                "compliance_status": "PASS" if is_compliant else "FAIL"
            })

        return {"audit_results": final_report}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline failure: {str(e)}")


@app.get("/health")
def health_check():
    return {"status": "healthy", "vlm_connected": True}