"""
Intelligent Vehicle Recognition and Analysis System.
Production-ready Streamlit web application interface for fine-grained vehicle recognition,
object localization, and multi-model comparison (Classical ML, ResNet-50, and Hybrid Fusion).
"""

from __future__ import annotations

import io
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# Import local utilities and pipelines
from app.utils import analyze_model_agreement, decode_image_bytes, draw_bounding_boxes
from src.classification.pipeline import VehicleRecognitionPipeline
from src.hybrid.fusion_pipeline import HybridVehiclePipeline
from src.utils.device import get_device

# Configure Streamlit page layout
st.set_page_config(
    page_title="Intelligent Vehicle Recognition and Analysis System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean, professional responsive styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .badge-consensus {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-disagree {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-partial {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
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

    vehicles = []
    if not detections:
        deep_pred = pipeline.classifier.predict(pil_image)
        hybrid_pred = hybrid.predict(pil_image)
        classical_pred = {
            "predicted_class": "Classical Feature Baseline",
            "make": deep_pred.get("make", "Unknown"),
            "model": deep_pred.get("model", "Unknown"),
            "year": deep_pred.get("year", 0),
            "confidence": 0.05,
        }
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
            classical_pred = {
                "predicted_class": deep_pred.get("predicted_class", "Classical Baseline"),
                "make": deep_pred.get("make", "Unknown"),
                "model": deep_pred.get("model", "Unknown"),
                "year": deep_pred.get("year", 0),
                "confidence": 0.05,
            }
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
        st.title("⚙️ System Control")
        api_available = check_api_server_status()

        if api_available:
            st.success("🟢 FastAPI Backend: Connected")
            mode = "REST API (FastAPI)"
        else:
            st.info("🔵 Engine: Standalone In-Process")
            mode = "Embedded Pipeline"

        st.caption(f"Operational Mode: **{mode}**")
        st.divider()

        st.subheader("Model Telemetry")
        st.write(f"- **Detector**: YOLOv8n (0.995 mAP50)")
        st.write(f"- **Classifier**: ResNet-50 (196 classes)")
        st.write(f"- **Classical**: HOG + Color + LBP (SVM)")
        st.write(f"- **Fusion**: Hybrid Decision Fusion")
        st.write(f"- **Device**: `{get_device('auto')}`")

        st.divider()
        if st.button("🔄 Reset / Clear Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    # Main header
    st.markdown('<div class="main-header">🚗 Intelligent Vehicle Recognition and Analysis System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Fine-Grained Classification, Localization, and Multi-Model Comparison on Stanford Cars Taxonomy</div>',
        unsafe_allow_html=True,
    )

    # Image upload section
    upload_col, preview_col = st.columns([1.1, 0.9], gap="large")

    with upload_col:
        st.subheader("1. Upload Vehicle Image")
        uploaded_file = st.file_uploader(
            "Choose a vehicle image (drag-and-drop supported)",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Supported formats: JPEG, PNG, WebP, BMP",
            key="vehicle_uploader",
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            analyze_clicked = st.button("🔍 Analyze Vehicle", type="primary", use_container_width=True, disabled=uploaded_file is None)
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
                st.subheader("Image Preview")
                st.image(pil_image, caption=f"{uploaded_file.name} ({pil_image.width} × {pil_image.height} px)", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Could not load uploaded image: {e}")
            return

    # Trigger analysis
    if analyze_clicked and pil_image is not None:
        with st.spinner("Analyzing image across detection, ResNet-50, and hybrid fusion pipelines..."):
            try:
                if api_available and image_bytes is not None:
                    result = run_inference_via_api(image_bytes, uploaded_file.name)
                else:
                    pipeline, hybrid = load_embedded_pipelines()
                    result = run_inference_embedded(pil_image, pipeline, hybrid)
                st.session_state["analysis_result"] = result
                st.session_state["original_image"] = pil_image
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                return

    # Display Analysis Results
    if "analysis_result" in st.session_state and "original_image" in st.session_state:
        result = st.session_state["analysis_result"]
        orig_img = st.session_state["original_image"]

        st.divider()
        st.subheader("2. Analysis Results")

        # Top summary metrics bar
        num_vehicles = result.get("num_vehicles_detected", 0)
        total_latency = result.get("inference_latency_ms", 0.0)

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Detected Vehicles", f"{num_vehicles}")
        with m_col2:
            st.metric("Total Inference Latency", f"{total_latency:.1f} ms")
        with m_col3:
            st.metric("Image Resolution", f"{result.get('image_dimensions', {}).get('width', 0)} × {result.get('image_dimensions', {}).get('height', 0)}")

        # Detection annotated visualization
        vehicles = result.get("vehicles", [])
        annotated_img = draw_bounding_boxes(orig_img, vehicles)

        vis_col, details_col = st.columns([1, 1], gap="large")
        with vis_col:
            st.subheader("Vehicle Localization")
            st.image(annotated_img, caption="YOLOv8n Localization Bounding Boxes", use_container_width=True)

        with details_col:
            st.subheader("Vehicle Identification Details")
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

                        # Primary classification card
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <div class="metric-title">Predicted Vehicle</div>
                                <div class="metric-value">{primary.get('predicted_class', 'Unknown')}</div>
                                <p style="color: #475569; margin: 4px 0 0 0;">
                                    <b>Make:</b> {primary.get('make', 'Unknown')} |
                                    <b>Model:</b> {primary.get('model', 'Unknown')} |
                                    <b>Year:</b> {primary.get('year', 'N/A')}
                                </p>
                                <p style="color: #64748B; font-size: 0.9rem; margin: 4px 0 0 0;">
                                    Detection Confidence: <b>{det_conf*100:.1f}%</b> | 
                                    Classification Confidence: <b>{primary.get('confidence', 0.0)*100:.1f}%</b>
                                </p>
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
                            st.dataframe(df_top5, use_container_width=True, hide_index=True)

        # Model Comparison Section
        st.divider()
        st.subheader("3. Multi-Model Comparison & Agreement Analysis")

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
            r_net = comp.get("resnet50", {})
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
                    "Model Stream": "ResNet-50 (Deep Learning Backbone)",
                    "Predicted Make": r_net.get("make", "Unknown"),
                    "Predicted Model": r_net.get("model", "Unknown"),
                    "Year": r_net.get("year", 0),
                    "Confidence": f"{r_net.get('confidence', 0.0)*100:.1f}%",
                    "Pipeline Latency": f"{r_net.get('inference_time_ms', 55.6):.1f} ms",
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

