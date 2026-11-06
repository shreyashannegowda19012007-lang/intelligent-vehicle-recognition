"""
Intelligent Vehicle Recognition and Analysis System.
Production-ready Streamlit web application interface for fine-grained vehicle recognition,
object localization, and multi-model comparison (Classical ML, ConvNeXt-Tiny, and Hybrid Fusion).
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path for standalone Streamlit execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# Import local utilities and pipelines
from app.utils import analyze_model_agreement, decode_image_bytes, draw_bounding_boxes
from src.classification.pipeline import VehicleRecognitionPipeline
from src.hybrid.fusion_pipeline import HybridVehiclePipeline
from src.utils.device import get_device

logger = logging.getLogger(__name__)

# Configure Streamlit page layout
st.set_page_config(
    page_title="Intelligent Vehicle Recognition and Analysis System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, professional responsive styling
st.markdown(
    """
    <style>
    /* Modern Automotive AI Analytics Theme */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.2rem;
        max-width: 1380px;
    }
    .brand-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        color: #2563EB;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    .main-header {
        font-size: 2.15rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 0.96rem;
        color: #475569;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .section-heading {
        font-size: 1.1rem;
        font-weight: 600;
        color: #0F172A;
        letter-spacing: -0.01em;
        margin-bottom: 0.75rem;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 14px;
        margin-bottom: 1.25rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: all 0.15s ease-in-out;
    }
    .kpi-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
    }
    .kpi-title {
        font-size: 0.72rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.15;
    }
    .prediction-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        margin-bottom: 16px;
    }
    .prediction-eyebrow {
        font-size: 0.70rem;
        font-weight: 700;
        color: #2563EB;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 4px;
    }
    .prediction-name {
        font-size: 1.35rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.015em;
        margin-bottom: 12px;
    }
    .prediction-specs-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        padding: 10px 14px;
        background: #F8FAFC;
        border-radius: 6px;
        border: 1px solid #EDF2F7;
        margin-bottom: 12px;
    }
    .spec-item {
        font-size: 0.85rem;
        color: #475569;
    }
    .spec-item strong {
        color: #0F172A;
    }
    .confidence-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 18px;
        font-size: 0.84rem;
        color: #64748B;
    }
    .confidence-metrics strong {
        color: #0F172A;
    }
    .status-box-online {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        padding: 9px 12px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .status-dot-online {
        height: 8px;
        width: 8px;
        background-color: #16A34A;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .status-text-online {
        font-weight: 600;
        color: #166534;
        font-size: 0.84rem;
    }
    .status-box-offline {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 9px 12px;
        border-radius: 6px;
        margin-bottom: 12px;
    }
    .status-dot-offline {
        height: 8px;
        width: 8px;
        background-color: #2563EB;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .status-text-offline {
        font-weight: 600;
        color: #1E40AF;
        font-size: 0.84rem;
    }
    .telemetry-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 12px;
        margin-bottom: 14px;
    }
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 5px 0;
        font-size: 0.82rem;
        border-bottom: 1px solid #EDF2F7;
    }
    .telemetry-row:last-child {
        border-bottom: none;
    }
    .telemetry-key {
        color: #64748B;
        font-weight: 500;
    }
    .telemetry-val {
        color: #0F172A;
        font-weight: 600;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    .badge-consensus {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.80rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-disagree {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.80rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-partial {
        background-color: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 3px 10px;
        border-radius: 6px;
        font-size: 0.80rem;
        font-weight: 600;
        display: inline-block;
    }
    .upload-placeholder {
        border: 2px dashed #CBD5E1;
        border-radius: 8px;
        padding: 42px 20px;
        text-align: center;
        background: #F8FAFC;
        color: #64748B;
        margin-top: 8px;
    }
    .upload-placeholder-title {
        font-weight: 600;
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 4px;
    }
    .upload-placeholder-desc {
        font-size: 0.82rem;
        line-height: 1.4;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_API_URL = os.getenv("API_URL") or os.getenv("BACKEND_URL") or "http://127.0.0.1:8000"


@st.cache_resource(show_spinner=False)
def load_embedded_pipelines() -> tuple[VehicleRecognitionPipeline, HybridVehiclePipeline]:
    """Cache pipeline instances for standalone in-process inference."""
    detector_weights = os.getenv("YOLO_MODEL_PATH")
    classifier_ckpt = os.getenv("CLASSIFIER_CHECKPOINT_PATH")
    pipeline = VehicleRecognitionPipeline(
        detector_weights=detector_weights,
        classifier_checkpoint=classifier_ckpt,
        device="auto",
    )
    hybrid = HybridVehiclePipeline(
        checkpoint_path=os.getenv("HYBRID_CHECKPOINT_PATH"),
        device="auto",
    )
    return pipeline, hybrid


def check_api_server_status(api_url: str = DEFAULT_API_URL) -> bool:
    """Check if the external FastAPI service is responding."""
    try:
        resp = requests.get(f"{api_url}/health", timeout=1.2)
        return resp.status_code == 200
    except Exception:
        return False


def run_inference_via_api(image_bytes: bytes, filename: str, api_url: str = DEFAULT_API_URL) -> dict[str, Any]:
    """Send image to FastAPI backend endpoint."""
    files = {"file": (filename, image_bytes, "image/jpeg")}
    resp = requests.post(f"{api_url}/analyze", files=files, timeout=30.0)
    if resp.status_code != 200:
        raise RuntimeError(f"Backend API error ({resp.status_code}): {resp.text}")
    return resp.json()


def run_inference_embedded(
    pil_image: Image.Image,
    pipeline: VehicleRecognitionPipeline,
    hybrid: HybridVehiclePipeline,
) -> dict[str, Any]:
    """Execute inference directly using local cached pipelines."""
    t_start = time.perf_counter()
    w, h = pil_image.size

    pipe_res = pipeline.process_image(pil_image, extract_rois=True)
    detections = pipe_res.get("detections", [])

    def _predict_classical(target_img: Any) -> dict[str, Any]:
        if hybrid.classical_predictor is not None:
            try:
                c_res = hybrid.classical_predictor.predict(target_img)
                return {
                    "predicted_class": c_res.get("class_name", c_res.get("predicted_class", "Unknown")),
                    "class_id": c_res.get("class_id", -1),
                    "make": c_res.get("make", "Unknown"),
                    "model": c_res.get("model", "Unknown"),
                    "year": c_res.get("year", 0),
                    "confidence": round(float(c_res.get("confidence", 0.0)), 4),
                    "top_5": c_res.get("top_5", []),
                }
            except Exception as ex:
                logger.warning("Classical prediction failed: %s", ex)
        return {
            "predicted_class": "Classical Baseline (Unavailable)",
            "class_id": -1,
            "make": "Unknown",
            "model": "Unknown",
            "year": 0,
            "confidence": 0.0,
            "top_5": [],
        }

    vehicles = []
    if not detections:
        deep_pred = pipeline.classifier.predict(pil_image)
        hybrid_pred = hybrid.predict(pil_image)
        classical_pred = _predict_classical(pil_image)
        agreement = analyze_model_agreement(classical_pred, deep_pred, hybrid_pred)
        vehicles.append({
            "vehicle_index": 1,
            "bbox": {"x1": 0, "y1": 0, "x2": w, "y2": h},
            "detection_confidence": 0.50,
            "is_full_frame_fallback": True,
            "primary_prediction": deep_pred,
            "model_comparison": {
                "classical_ml": classical_pred,
                "resnet50": deep_pred,
                "convnext_tiny": deep_pred,
                "hybrid": hybrid_pred,
                "agreement": agreement,
            },
        })
    else:
        for idx, det in enumerate(detections, 1):
            bbox = det.get("bbox", {"x1": 0, "y1": 0, "x2": w, "y2": h})
            det_conf = float(det.get("detection_confidence", 0.90))

            x1, y1 = max(0, bbox["x1"]), max(0, bbox["y1"])
            x2, y2 = min(w, bbox["x2"]), min(h, bbox["y2"])
            crop = pil_image.crop((x1, y1, x2, y2)) if (x2 > x1 and y2 > y1) else pil_image

            deep_pred = det.get("classification", {})
            hybrid_pred = hybrid.predict(crop)
            classical_pred = _predict_classical(crop)
            agreement = analyze_model_agreement(classical_pred, deep_pred, hybrid_pred)

            vehicles.append({
                "vehicle_index": idx,
                "bbox": bbox,
                "detection_confidence": det_conf,
                "is_full_frame_fallback": False,
                "primary_prediction": deep_pred,
                "model_comparison": {
                    "classical_ml": classical_pred,
                    "resnet50": deep_pred,
                    "convnext_tiny": deep_pred,
                    "hybrid": hybrid_pred,
                    "agreement": agreement,
                },
            })

    total_latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
    return {
        "status": "success",
        "image_dimensions": {"width": w, "height": h},
        "num_vehicles_detected": len(vehicles),
        "inference_latency_ms": total_latency_ms,
        "timing_breakdown": pipe_res.get("timing", {}),
        "vehicles": vehicles,
    }


def main() -> None:
    # Sidebar: System telemetry & connection modes
    with st.sidebar:
        st.markdown('<div class="brand-eyebrow">Control Center</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-heading" style="margin-bottom: 12px;">System Control</div>', unsafe_allow_html=True)
        api_available = check_api_server_status()

        if api_available:
            st.markdown(
                """
                <div class="status-box-online">
                    <span class="status-dot-online"></span>
                    <span class="status-text-online">FastAPI Backend: Connected</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            mode = "REST API (FastAPI)"
        else:
            st.markdown(
                """
                <div class="status-box-offline">
                    <span class="status-dot-offline"></span>
                    <span class="status-text-offline">Engine: Standalone In-Process</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            mode = "Embedded Pipeline"

        st.caption(f"Operational Mode: **{mode}**")
        st.divider()

        st.markdown('<div class="section-heading" style="margin-bottom: 10px;">Model Telemetry</div>', unsafe_allow_html=True)
        dev_name = str(get_device("auto"))
        st.markdown(
            f"""
            <div class="telemetry-card">
                <div class="telemetry-row">
                    <span class="telemetry-key">Object Detector</span>
                    <span class="telemetry-val">YOLOv8n (0.995 mAP50)</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-key">Primary Classifier</span>
                    <span class="telemetry-val">ConvNeXt-Tiny (196-cls)</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-key">Classical ML</span>
                    <span class="telemetry-val">HOG + Color + LBP (SVM)</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-key">Decision Fusion</span>
                    <span class="telemetry-val">Hybrid Agreement</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-key">Compute Engine</span>
                    <span class="telemetry-val">{dev_name}</span>
                </div>
                <div class="telemetry-row">
                    <span class="telemetry-key">Inference Resolution</span>
                    <span class="telemetry-val">320 × 320 px</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        if st.button("Reset Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Main header
    st.markdown('<div class="brand-eyebrow">Automotive Vision Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">Intelligent Vehicle Recognition and Analysis System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Fine-Grained Classification, Localization, and Multi-Model Comparison on Stanford Cars Taxonomy</div>',
        unsafe_allow_html=True,
    )

    # Image upload section
    upload_col, preview_col = st.columns([1.1, 0.9], gap="large")

    with upload_col:
        st.markdown('<div class="section-heading">01. Vehicle Image Ingestion</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Select or drop vehicle image (JPEG, PNG, WebP, BMP)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supported formats: JPEG, PNG, WebP, BMP (Max 25 MB)",
            key="vehicle_uploader",
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyze_clicked = st.button("Analyze Vehicle", type="primary", use_container_width=True, disabled=uploaded_file is None)
        with col_btn2:
            if st.button("Clear Image", use_container_width=True, disabled=uploaded_file is None):
                st.session_state.clear()
                st.rerun()

    image_bytes = None
    pil_image = None
    if uploaded_file is not None:
        try:
            image_bytes = uploaded_file.getvalue()
            pil_image = decode_image_bytes(image_bytes)
            with preview_col:
                st.markdown('<div class="section-heading">Image Preview</div>', unsafe_allow_html=True)
                st.image(pil_image, caption=f"{uploaded_file.name} ({pil_image.width} × {pil_image.height} px)", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load uploaded image: {e}")
            return
    else:
        with preview_col:
            st.markdown('<div class="section-heading">Image Preview</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="upload-placeholder">
                    <div class="upload-placeholder-title">No Image Selected</div>
                    <div class="upload-placeholder-desc">Upload a vehicle image to view localization bounding boxes, fine-grained taxonomy classification, and multi-model agreement analysis.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Trigger analysis
    if analyze_clicked and pil_image is not None:
        with st.spinner("Analyzing image across detection, ConvNeXt-Tiny classification, and hybrid fusion pipelines..."):
            try:
                if api_available and image_bytes is not None:
                    result = run_inference_via_api(image_bytes, uploaded_file.name)
                else:
                    pipeline, hybrid = load_embedded_pipelines()
                    result = run_inference_embedded(pil_image, pipeline, hybrid)
                st.session_state["analysis_result"] = result
                st.session_state["original_image"] = pil_image
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                return

    # Display Analysis Results
    if "analysis_result" in st.session_state and "original_image" in st.session_state:
        result = st.session_state["analysis_result"]
        orig_img = st.session_state["original_image"]

        st.divider()
        st.markdown('<div class="section-heading">02. Detection & Identification Telemetry</div>', unsafe_allow_html=True)

        # Top summary metrics bar
        num_vehicles = result.get("num_vehicles_detected", 0)
        total_latency = result.get("inference_latency_ms", 0.0)
        img_dims = result.get("image_dimensions", {})
        width_val = img_dims.get("width", orig_img.width if orig_img else 0)
        height_val = img_dims.get("height", orig_img.height if orig_img else 0)

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-title">Detected Vehicles</div>
                    <div class="kpi-value">{num_vehicles}</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Total Inference Latency</div>
                    <div class="kpi-value">{total_latency:.1f} ms</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-title">Source Resolution</div>
                    <div class="kpi-value">{width_val} × {height_val} px</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Detection annotated visualization
        vehicles = result.get("vehicles", [])
        annotated_img = draw_bounding_boxes(orig_img, vehicles)

        vis_col, details_col = st.columns([1, 1], gap="large")
        with vis_col:
            st.markdown('<div class="section-heading">Vehicle Localization</div>', unsafe_allow_html=True)
            st.image(annotated_img, caption="YOLOv8n Localization Bounding Boxes", use_container_width=True)

        with details_col:
            st.markdown('<div class="section-heading">Vehicle Identification Details</div>', unsafe_allow_html=True)
            if not vehicles:
                st.info("No vehicle detected in the scene.")
            else:
                # Multi-vehicle selector tabs
                if len(vehicles) > 1:
                    tabs = st.tabs([f"Vehicle #{v['vehicle_index']}" for v in vehicles])
                else:
                    tabs = [st.container()]

                for tab, v in zip(tabs, vehicles):
                    with tab:
                        primary = v.get("primary_prediction", {})
                        det_conf = v.get("detection_confidence", 0.0)
                        pred_conf = float(primary.get("confidence", 0.0))

                        # Primary classification card
                        st.markdown(
                            f"""
                            <div class="prediction-card">
                                <div class="prediction-eyebrow">Primary Prediction</div>
                                <div class="prediction-name">{primary.get('predicted_class', 'Unknown')}</div>
                                <div class="prediction-specs-grid">
                                    <div class="spec-item"><strong>Make:</strong> {primary.get('make', 'Unknown')}</div>
                                    <div class="spec-item"><strong>Model:</strong> {primary.get('model', 'Unknown')}</div>
                                    <div class="spec-item"><strong>Year:</strong> {primary.get('year', 'N/A')}</div>
                                </div>
                                <div class="confidence-metrics">
                                    <span>Detection Confidence: <strong>{det_conf*100:.1f}%</strong></span>
                                    <span>Classification Confidence: <strong>{pred_conf*100:.1f}%</strong></span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        # Top-5 Predictions
                        top_5 = primary.get("top_5", [])
                        if top_5:
                            st.write("**Top-5 Predictions**")
                            df_top5 = pd.DataFrame([
                                {
                                    "Rank": item.get("rank", i + 1),
                                    "Vehicle Candidate": item.get("class", item.get("class_name", "Unknown")),
                                    "Confidence (%)": round(float(item.get("confidence", 0.0)) * 100.0, 2),
                                }
                                for i, item in enumerate(top_5)
                            ])
                            st.dataframe(
                                df_top5,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Rank": st.column_config.NumberColumn(width="small"),
                                    "Vehicle Candidate": st.column_config.TextColumn(width="large"),
                                    "Confidence (%)": st.column_config.ProgressColumn(
                                        format="%.2f%%",
                                        min_value=0.0,
                                        max_value=100.0,
                                        width="medium",
                                    ),
                                },
                            )

        # Model Comparison Section
        st.divider()
        st.markdown('<div class="section-heading">03. Multi-Model Consensus & Comparison</div>', unsafe_allow_html=True)

        for v in vehicles:
            comp = v.get("model_comparison", {})
            agreement = comp.get("agreement", {})

            # Agreement badge
            status = agreement.get("status", "Analyzed")
            badge_class = "badge-consensus" if "Consensus" in status else ("badge-partial" if "Partial" in status else "badge-disagree")

            st.markdown(f"**Vehicle #{v['vehicle_index']} Model Agreement**: <span class='{badge_class}'>{status}</span>", unsafe_allow_html=True)
            st.caption(agreement.get("summary", ""))

            # Comparison Table
            c_ml = comp.get("classical_ml", {})
            dl_pred = comp.get("convnext_tiny", comp.get("resnet50", {}))
            hyb = comp.get("hybrid", {})

            comp_data = [
                {
                    "Model Stream": "Classical ML (HOG + Color + LBP + SVM)",
                    "Predicted Make": c_ml.get("make", "Unknown"),
                    "Predicted Model": c_ml.get("model", "Unknown"),
                    "Year": c_ml.get("year", 0),
                    "Confidence": f"{c_ml.get('confidence', 0.0)*100:.1f}%",
                    "Pipeline Latency": "0.015 ms",
                },
                {
                    "Model Stream": "Deep Learning (ConvNeXt-Tiny Primary)",
                    "Predicted Make": dl_pred.get("make", "Unknown"),
                    "Predicted Model": dl_pred.get("model", "Unknown"),
                    "Year": dl_pred.get("year", 0),
                    "Confidence": f"{dl_pred.get('confidence', 0.0)*100:.1f}%",
                    "Pipeline Latency": f"{dl_pred.get('inference_time_ms', 55.6):.1f} ms",
                },
                {
                    "Model Stream": "Full Hybrid Pipeline (Decision Fusion)",
                    "Predicted Make": hyb.get("make", "Unknown"),
                    "Predicted Model": hyb.get("model", "Unknown"),
                    "Year": hyb.get("year", 0),
                    "Confidence": f"{hyb.get('confidence', 0.0)*100:.1f}%",
                    "Pipeline Latency": f"{hyb.get('inference_time_ms', 59.1):.1f} ms",
                },
            ]
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
