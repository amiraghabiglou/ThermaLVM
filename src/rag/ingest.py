import os
import fitz  # PyMuPDF
import lancedb
from lancedb.pydantic import Vector, LanceModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. Configuration & Schema
# ---------------------------------------------------------
DB_PATH = os.getenv("DB_PATH", "./data/vector_store")
PDF_DIR = os.getenv("PDF_DIR", "./data/regulations")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize the embedding model locally (CPU/Edge friendly)
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
VECTOR_DIM = encoder.get_sentence_embedding_dimension()


class RegulationChunk(LanceModel):
    vector: Vector(VECTOR_DIM)  # type: ignore
    text_content: str
    code_ref: str
    defect_type: str
    max_allowed_delta_t: float


# ---------------------------------------------------------
# 2. Parsing & Transformation Logic
# ---------------------------------------------------------
# In a full production system, an LLM would extract these mappings offline.
# For Phase 1, we use a deterministic rule-engine to map text to our API contract.
DEFECT_MAPPING_RULES = {
    "roof": ("roof_heat_loss", 5.0),
    "window": ("window_seal_failure", 3.5),
    "insulation": ("wall_insulation_gap", 4.0),
    "hvac": ("hvac_exhaust_anomaly", 10.0),
    "bridge": ("thermal_bridge", 2.0)
}


def parse_and_chunk_pdf(pdf_path: str) -> list[dict]:
    """Extracts text from a PDF and chunks it hierarchically by paragraph."""
    doc = fitz.open(pdf_path)
    chunks = []

    doc_name = os.path.basename(pdf_path).replace(".pdf", "")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Simple hierarchical chunking by double line breaks (paragraphs/sections)
        paragraphs = text.split("\n\n")

        for i, para in enumerate(paragraphs):
            clean_text = para.strip().replace("\n", " ")
            if len(clean_text) < 50:  # Skip useless headers/footers
                continue

            # Determine defect type and threshold via heuristic mapping
            mapped_defect = "unknown"
            threshold = 999.0
            for keyword, (defect_enum, max_delta) in DEFECT_MAPPING_RULES.items():
                if keyword in clean_text.lower():
                    mapped_defect = defect_enum
                    threshold = max_delta
                    break

            # Only store chunks relevant to our energy audit
            if mapped_defect != "unknown":
                chunks.append({
                    "text_content": clean_text,
                    "code_ref": f"{doc_name} - Sec {page_num + 1}.{i}",
                    "defect_type": mapped_defect,
                    "max_allowed_delta_t": threshold
                })

    return chunks


# ---------------------------------------------------------
# 3. Database Ingestion
# ---------------------------------------------------------
def ingest_regulations():
    """Main pipeline: Parse -> Embed -> Insert into LanceDB."""
    db = lancedb.connect(DB_PATH)

    # Drop existing table if we are rebuilding the knowledge base
    if "regulations" in db.table_names():
        db.drop_table("regulations")

    table = db.create_table("regulations", schema=RegulationChunk)

    all_records = []

    if not os.path.exists(PDF_DIR):
        print(f"Directory {PDF_DIR} not found. Please add regulatory PDFs.")
        return

    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            file_path = os.path.join(PDF_DIR, filename)
            print(f"Processing {filename}...")

            chunks = parse_and_chunk_pdf(file_path)

            # Batch encode the text chunks
            texts = [c["text_content"] for c in chunks]
            if not texts:
                continue

            embeddings = encoder.encode(texts)

            # Merge embeddings with metadata
            for chunk, emb in zip(chunks, embeddings):
                chunk["vector"] = emb.tolist()
                all_records.append(chunk)

    if all_records:
        table.add(all_records)
        print(f"Successfully ingested {len(all_records)} regulatory chunks into LanceDB.")
    else:
        print("No valid regulatory text found to ingest.")


if __name__ == "__main__":
    ingest_regulations()