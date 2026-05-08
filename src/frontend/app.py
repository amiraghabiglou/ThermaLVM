import os
import io
import requests
import pandas as pd
import gradio as gr
from PIL import Image, ImageDraw

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ---------------------------------------------------------
# 2. Visualization Logic
# ---------------------------------------------------------
def draw_bounding_boxes(image: Image.Image, defects: list) -> Image.Image:
    """Draws bounding boxes and labels on the image based on normalized coordinates."""
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img)
    width, height = annotated_img.size

    for defect in defects:
        box = defect["bounding_box"]
        # Convert normalized coordinates (0.0 - 1.0) to absolute pixels
        x_min = int(box["x_min"] * width)
        y_min = int(box["y_min"] * height)
        x_max = int(box["x_max"] * width)
        y_max = int(box["y_max"] * height)

        status = defect["compliance_status"]
        color = "red" if status == "FAIL" else "orange"

        # Draw the rectangle
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=4)

        # Draw the label
        label = f"{defect['defect']} (ΔT: {defect['detected_delta_t']}°C) - {status}"

        # Optional: Add a background rectangle for text readability
        # text_bbox = draw.textbbox((x_min, y_min - 15), label)
        # draw.rectangle(text_bbox, fill=color)
        draw.text((x_min + 5, y_min - 15), label, fill=color)

    return annotated_img


# ---------------------------------------------------------
# 3. API Communication
# ---------------------------------------------------------
def run_audit(rgb_img: Image.Image, thermal_img: Image.Image):
    """Sends images to the FastAPI backend and processes the results."""
    if rgb_img is None or thermal_img is None:
        return None, pd.DataFrame([{"Error": "Both RGB and Thermal images are required."}])

    # Resize images to reduce token count and drastically speed up VLM inference
    max_size = (1024, 1024)
    rgb_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    thermal_img.thumbnail(max_size, Image.Resampling.LANCZOS)
    # Convert PIL Images to bytes for the multipart/form-data request
    rgb_byte_arr = io.BytesIO()
    rgb_img.save(rgb_byte_arr, format='JPEG')
    rgb_bytes = rgb_byte_arr.getvalue()

    thermal_byte_arr = io.BytesIO()
    thermal_img.save(thermal_byte_arr, format='JPEG')
    thermal_bytes = thermal_byte_arr.getvalue()

    try:
        response = requests.post(
            f"{API_URL}/audit",
            files={
                "rgb_image": ("rgb.jpg", rgb_bytes, "image/jpeg"),
                "thermal_image": ("thermal.jpg", thermal_bytes, "image/jpeg")
            },
            timeout=300  # VLM inference can take a moment
        )

        if response.status_code != 200:
            return rgb_img, pd.DataFrame([{"Error": f"API Error {response.status_code}: {response.text}"}])

        data = response.json()
        results = data.get("audit_results", [])

        if not results:
            return rgb_img, pd.DataFrame([{"Result": "No thermal defects detected."}])

        # 1. Annotate the RGB image for visual reference
        annotated_rgb = draw_bounding_boxes(rgb_img, results)

        # 2. Format the compliance report for the data table
        df_report = pd.DataFrame([{
            "Defect Type": r["defect"],
            "Detected ΔT (°C)": r["detected_delta_t"],
            "Allowed ΔT (°C)": r["regulation_threshold"],
            "Regulation Code": r["regulation_code"],
            "Status": r["compliance_status"]
        } for r in results])

        return annotated_rgb, df_report

    except requests.exceptions.ConnectionError:
        return rgb_img, pd.DataFrame([{"Error": f"Failed to connect to API at {API_URL}. Is the backend running?"}])
    except Exception as e:
        return rgb_img, pd.DataFrame([{"Error": f"Unexpected error: {str(e)}"}])


# ---------------------------------------------------------
# 4. Gradio Interface Layout
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Default(primary_hue="blue", secondary_hue="gray")) as app:
    gr.Markdown("# 🏭 ThermaLVM: Edge-Deployable Industrial Energy Audit")
    gr.Markdown(
        "Upload synchronized RGB and Thermal images of a facility. The system uses a Vision-Language Model to detect anomalies and cross-references them with local building codes to generate a compliance report.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input Images")
            in_rgb = gr.Image(type="pil", label="RGB Image (High Resolution)")
            in_thermal = gr.Image(type="pil", label="Thermal Image (Radiometric)")
            submit_btn = gr.Button("Run Thermal Audit", variant="primary")

        with gr.Column():
            gr.Markdown("### Audit Results")
            out_img = gr.Image(type="pil", label="Annotated Defect Map", interactive=False)
            out_table = gr.Dataframe(label="Regulatory Compliance Report", interactive=False)

    submit_btn.click(
        fn=run_audit,
        inputs=[in_rgb, in_thermal],
        outputs=[out_img, out_table]
    )

# if __name__ == "__main__":
#     app.launch(
#         server_name="0.0.0.0",
#         server_port=8502,
#         share=False,
#         show_error=True
#     )
if __name__ == "__main__":
    app.launch(share=False, show_error=True)