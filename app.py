import os
import io
import time
from datetime import datetime
from pathlib import Path
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline import (
    load_image,
    get_image_metadata,
    preprocess,
    analyze,
    detect,
    segment,
    segmentation_ok,
    classify,
    surrogate_heatmap,
    gradcam,          # alias kept for backward compat
    tumor_area,
    extract_features,
    get_prediction_breakdown
)

# ============================================================================
# 1. Page Configuration & Custom Medical Dashboard Theme
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="Brain Tumor Detection",
    page_icon="🧠"
)

# Custom Styling for Academic & Medical Polish
st.markdown("""
<style>
    /* Global & Typography */
    [data-testid="stMainBlockContainer"],
    .stMainBlockContainer,
    .block-container {
        padding-top: 5.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 100% !important;
    }
    
    /* Medical Cards */
    .medical-card {
        background-color: #FFFFFF;
        border: 1px solid #E2D9CF;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 12px rgba(31, 42, 68, 0.04);
    }
    
    /* Streamlit Native Metric Card Overrides */
    [data-testid="stMetric"] {
        background-color: #FBF3EA;
        border: 1px solid #E2D9CF;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 8px rgba(31, 42, 68, 0.03);
    }
    [data-testid="stMetricLabel"] {
        color: #556B82 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #1F2A44 !important;
        font-family: serif !important;
        font-weight: 700 !important;
    }
    
    /* Pipeline Step Tracker */
    .pipeline-tracker-box {
        background-color: #FFFFFF;
        border: 1px solid #E2D9CF;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(31, 42, 68, 0.04);
    }
    .pipeline-tracker-header {
        font-weight: 700;
        font-size: 0.95rem;
        color: #1F2A44;
        margin-bottom: 10px;
        font-family: serif;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .pipeline-steps-wrapper {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
    }
    .step-pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 5px 13px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
        transition: all 0.2s ease-in-out;
    }
    .step-completed {
        background-color: #1F2A44;
        color: #FFFFFF;
        border: 1px solid #1F2A44;
    }
    .step-active {
        background-color: #FBF3EA;
        color: #1F2A44;
        border: 1px solid #1F2A44;
        font-weight: 700;
    }
    .step-pending {
        background-color: #F8FAFC;
        color: #94A3B8;
        border: 1px solid #E2E8F0;
    }
    .step-arrow {
        font-size: 0.85rem;
        font-weight: bold;
    }
    .arrow-completed {
        color: #1F2A44;
    }
    .arrow-pending {
        color: #CBD5E1;
    }

    /* Empty State Box */
    .empty-state-box {
        background-color: #FBF3EA;
        border: 1px dashed #1F2A44;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 2px 8px rgba(31, 42, 68, 0.03);
    }
    .empty-state-icon {
        font-size: 3rem;
        margin-bottom: 0.75rem;
    }
    .empty-state-title {
        font-family: serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #1F2A44;
        margin-bottom: 0.5rem;
    }
    .empty-state-text {
        font-size: 1rem;
        color: #4A5568;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Responsive image scaling */
    img {
        max-width: 100% !important;
        height: auto !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. Caching Wrappers for Pipeline Functions (Performance Optimization)
# ============================================================================
@st.cache_data(show_spinner=False)
def cached_preprocess(img_array: np.ndarray) -> np.ndarray:
    return preprocess(img_array)

@st.cache_data(show_spinner=False)
def cached_analyze(img_array: np.ndarray) -> dict:
    return analyze(img_array)

@st.cache_data(show_spinner=False)
def cached_segment(img_array: np.ndarray) -> np.ndarray:
    return segment(img_array)

@st.cache_data(show_spinner=False)
def cached_extract_features(img_array: np.ndarray, mask: np.ndarray) -> dict:
    return extract_features(img_array, mask)

@st.cache_data(show_spinner=False)
def cached_heatmap(img_array: np.ndarray) -> np.ndarray:
    return surrogate_heatmap(img_array)


# ============================================================================
# 3. Utility Functions: Overlay & Report Generation
# ============================================================================
def generate_tumor_overlay(base_img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Renders a semi-transparent crimson overlay over the segmented tumor
    region on the preprocessed MRI, delineated by a distinct contour outline.
    """
    overlay = base_img.copy()
    red_layer = np.zeros_like(base_img)
    red_layer[mask > 0] = [230, 40, 40]  # Crimson red

    # Alpha blending on positive mask pixels
    blended = np.where(
        mask[:, :, None] > 0,
        cv2.addWeighted(base_img, 1.0 - alpha, red_layer, alpha, 0),
        base_img
    )

    # Highlight boundary contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 45, 45), 2)
    return blended


def generate_report_png(
    input_img: np.ndarray,
    overlay_img: np.ndarray,
    heatmap_img: np.ndarray,
    tumor_type: str,
    confidence: float,
    area_info: dict,
    pixel_spacing: float,
    filename: str,
    seg_reliable: bool = True,
    uncertain: bool = False
) -> bytes:
    """
    Generates a high-resolution clinical diagnostic summary report exportable as PNG.
    """
    fig = plt.figure(figsize=(13, 9.5), facecolor="#FFFFFF", dpi=160)
    
    # 1. Header Banner
    fig.text(0.06, 0.95, "BRAIN TUMOR MRI DIAGNOSTIC REPORT", fontsize=18, fontweight="bold", color="#1F2A44", fontfamily="serif")
    fig.text(0.06, 0.92, "ABES Engineering College • Dept. of CSE • Session 2026-27 | Mini Project (BCS554)", fontsize=11, color="#556B82", fontfamily="serif")
    fig.text(0.06, 0.89, f"Target Scan: {filename}   |   Pixel Spacing: {pixel_spacing:.2f} mm   |   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9.5, color="#7B8A9E")
    
    # Dividing line
    line = plt.Line2D([0.06, 0.94], [0.87, 0.87], transform=fig.transFigure, color="#E2D9CF", linewidth=1.5)
    fig.add_artist(line)

    # 2. Three Image Panels
    ax1 = fig.add_axes([0.06, 0.48, 0.27, 0.35])
    ax1.imshow(input_img)
    ax1.set_title("1. Input MRI Scan", fontsize=12, fontweight="bold", color="#1F2A44", pad=8)
    ax1.axis("off")

    ax2 = fig.add_axes([0.365, 0.48, 0.27, 0.35])
    ax2.imshow(overlay_img)
    ax2.set_title("2. Segmented Tumor (Red Overlay)", fontsize=12, fontweight="bold", color="#B91C1C", pad=8)
    ax2.axis("off")

    ax3 = fig.add_axes([0.67, 0.48, 0.27, 0.35])
    ax3.imshow(heatmap_img)
    ax3.set_title("3. Surrogate Activation Heatmap", fontsize=12, fontweight="bold", color="#1F2A44", pad=8)
    ax3.axis("off")

    # 3. Diagnostic Summary Card
    rect = plt.Rectangle((0.06, 0.20), 0.88, 0.24, transform=fig.transFigure,
                         facecolor="#FBF3EA", edgecolor="#E2D9CF", linewidth=1.5, clip_on=False)
    fig.add_artist(rect)

    fig.text(0.09, 0.39, "CLINICAL FINDINGS & QUANTITATIVE ANALYSIS", fontsize=12, fontweight="bold", color="#1F2A44", fontfamily="serif")
    
    col_x1, col_x2, col_x3 = 0.09, 0.39, 0.69
    fig.text(col_x1, 0.33, "Condition / Status:", fontsize=10, color="#64748B")
    status_text = "TUMOR DETECTED (UNCERTAIN)" if uncertain else "TUMOR DETECTED"
    fig.text(col_x1, 0.29, status_text, fontsize=13, fontweight="bold", color="#B91C1C")
    
    fig.text(col_x2, 0.33, "Classified Subtype:", fontsize=10, color="#64748B")
    fig.text(col_x2, 0.29, f"{tumor_type} ({confidence*100:.1f}% conf.)", fontsize=13, fontweight="bold", color="#1F2A44")

    fig.text(col_x3, 0.33, "Cross-Sectional Area (this slice):", fontsize=10, color="#64748B")
    if seg_reliable:
        fig.text(col_x3, 0.29, f"{area_info['area_value']:.2f} {area_info['unit']}", fontsize=13, fontweight="bold", color="#1F2A44")
    else:
        fig.text(col_x3, 0.29, "Segmentation unreliable", fontsize=13, fontweight="bold", color="#94A3B8")

    fig.text(0.09, 0.23, "Notice: Decision-support aid only. Final diagnosis rests with a qualified medical doctor.",
             fontsize=9, fontstyle="italic", color="#64748B")

    # 4. Limitations Section
    lim_rect = plt.Rectangle((0.06, 0.03), 0.88, 0.14, transform=fig.transFigure,
                              facecolor="#FFF8F0", edgecolor="#E2D9CF", linewidth=1.0, clip_on=False)
    fig.add_artist(lim_rect)
    fig.text(0.09, 0.14, "⚠️ KNOWN LIMITATIONS", fontsize=10, fontweight="bold", color="#B45309", fontfamily="serif")
    limitations = [
        "• Segmentation is heuristic (intensity thresholding, NOT a trained U-Net). Ring-enhancing lesions may be partially missed.",
        "• Heatmap is a surrogate (Gaussian blob at segmentation centroid), NOT a true Grad-CAM or Eigen-CAM.",
        "• Training dataset may contain patient-level leakage and shortcut features (domain shift risk on unseen scanners).",
        "• Area measurement is from a single 2D slice — not volumetric. Uncalibrated unless DICOM pixel spacing is provided.",
    ]
    for i, lim in enumerate(limitations):
        fig.text(0.09, 0.11 - i * 0.025, lim, fontsize=7.5, color="#78716C")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# Pipeline steps sequence definition
PIPELINE_STEPS = [
    "Input MRI",
    "Preprocess",
    "Classify",
    "Segment",
    "Extract Features",
    "Measure Area",
]

def render_pipeline_tracker(completed_steps: list, active_step: str = None) -> str:
    """
    Renders an interactive, responsive HTML step indicator for the diagnostic pipeline.
    """
    html_parts = []
    html_parts.append('<div class="pipeline-tracker-box">')
    html_parts.append('<div class="pipeline-tracker-header">🩺 Diagnostic Pipeline Progress</div>')
    html_parts.append('<div class="pipeline-steps-wrapper">')
    
    for i, step in enumerate(PIPELINE_STEPS):
        if step in completed_steps:
            status_class = "step-completed"
            icon = "✓"
        elif step == active_step:
            status_class = "step-active"
            icon = "⏳"
        else:
            status_class = "step-pending"
            icon = "○"
            
        html_parts.append(f'<div class="step-pill {status_class}"><span>{icon}</span> {step}</div>')
        
        if i < len(PIPELINE_STEPS) - 1:
            arrow_class = "arrow-completed" if (step in completed_steps and PIPELINE_STEPS[i+1] in completed_steps) else "arrow-pending"
            html_parts.append(f'<span class="step-arrow {arrow_class}">➔</span>')
            
    html_parts.append('</div></div>')
    return "".join(html_parts)


# ============================================================================
# 4. Header Row & Institutional Branding
# ============================================================================
st.markdown('<div style="margin-top: 2.2rem;"></div>', unsafe_allow_html=True)
header_col1, header_col2 = st.columns([5, 1])

with header_col1:
    st.caption("MINI PROJECT • 5TH SEMESTER • BCS554")
    st.title("Detection of Brain Tumor from MRI Images")
    st.caption("ABES Engineering College, Ghaziabad | Dept. of CSE | Session 2026-27")

with header_col2:
    logo_path = Path(__file__).parent / "assets" / "abes_logo.png"
    if not logo_path.exists():
        logo_path = Path("assets/abes_logo.png")

    if logo_path.exists():
        try:
            st.image(str(logo_path), width=120)
        except Exception:
            pass  # Gracefully skip if logo file is invalid

# Clinical Guidance Disclaimer Banner
st.info("ℹ️ Decision-support aid only. Final diagnosis rests with a qualified doctor.")


# ============================================================================
# 5. Sidebar Controls & Configuration
# ============================================================================
with st.sidebar:
    st.header("Upload MRI")
    uploaded_file = st.file_uploader(
        "Upload brain MRI scan",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    st.subheader("Configuration")
    pixel_spacing = st.number_input(
        "Pixel spacing (mm)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.2f",
        help="In-plane spatial resolution for MRI slice (from DICOM metadata)"
    )

    show_heatmap = st.checkbox(
        "Show Activation Heatmap",
        value=True,
        help="Generate a surrogate activation heatmap centred on the segmentation region"
    )

    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

    st.divider()

    with st.expander("About", expanded=False):
        st.markdown("**Project ID:** `PJ_ID_2026_CSE_62`")
        st.markdown("**Supervisor:** Mr. Vivek Chaudhary")
        st.markdown("---")
        st.markdown("**Team Members:**")
        st.markdown("• **Ranjan Yadav** (2400320100889)")
        st.markdown("• **Rishabh Dubey** (2400320100900)")
        st.markdown("• **Rishabh Gupta** (2400320100901)")


# ============================================================================
# 6. Main Content Area & Execution Flow
# ============================================================================

# If no file is uploaded: show friendly empty state
if uploaded_file is None:
    st.session_state["analyzed_file"] = None
    st.markdown("""
    <div class="empty-state-box">
        <div class="empty-state-icon">🔬</div>
        <div class="empty-state-title">No MRI Scan Uploaded</div>
        <div class="empty-state-text">
            Please upload a brain MRI scan (JPG, JPEG, or PNG) from the sidebar to begin processing and evaluation.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Track analysis state so UI tabs and controls remain persistent
    if analyze_btn:
        st.session_state["analyzed_file"] = uploaded_file.name

    is_analyzed = (st.session_state.get("analyzed_file") == uploaded_file.name)

    if not is_analyzed:
        # Prompt user to click Analyze
        st.subheader("MRI Scan Preview")
        preview_img = load_image(uploaded_file)
        if preview_img:
            subcol1, subcol2 = st.columns([1, 1])
            with subcol1:
                st.image(preview_img, caption=f"Uploaded File: {uploaded_file.name}", width="stretch")
            with subcol2:
                meta = get_image_metadata(preview_img)
                st.markdown("#### Scan Metadata")
                st.markdown(f"- **Filename:** `{uploaded_file.name}`")
                st.markdown(f"- **Dimensions:** `{meta['dimensions']}`")
                st.markdown(f"- **Pixel Spacing:** `{pixel_spacing:.2f} mm`")
                st.markdown(f"- **Heatmap Enabled:** `{'Yes' if show_heatmap else 'No'}`")
                st.info("👆 Click **Analyze** in the sidebar to run the detection pipeline.")
        else:
            st.warning("⚠️ Could not load the uploaded file. Please verify it is a valid image.")

    else:
        # Pipeline execution with step tracker, progress spinner & error handling
        pipeline_placeholder = st.empty()
        completed_steps = []

        with st.spinner("Running pipeline..."):
            try:
                # Step 1: Input MRI
                pipeline_placeholder.markdown(
                    render_pipeline_tracker(completed_steps, active_step="Input MRI"),
                    unsafe_allow_html=True
                )
                raw_pil = load_image(uploaded_file)
                if raw_pil is None:
                    raise ValueError("Could not decode the uploaded image file.")
                input_rgb = np.array(raw_pil)
                completed_steps.append("Input MRI")

                # Step 2: Preprocess
                pipeline_placeholder.markdown(
                    render_pipeline_tracker(completed_steps, active_step="Preprocess"),
                    unsafe_allow_html=True
                )
                preprocessed_rgb = cached_preprocess(input_rgb)
                completed_steps.append("Preprocess")

                # Step 3: Classify (single inference via analyze())
                pipeline_placeholder.markdown(
                    render_pipeline_tracker(completed_steps, active_step="Classify"),
                    unsafe_allow_html=True
                )
                result = cached_analyze(preprocessed_rgb)
                probs = result["probs"]
                top_class = result["top"]
                is_uncertain = result["uncertain"]
                tumor_prob = result["tumor_prob"]
                tumor_detected = top_class != "No Tumor"

                if not tumor_detected:
                    # Healthy / Negative Scan Workflow
                    det_conf = probs["No Tumor"]
                    completed_steps.extend(["Classify", "Segment", "Extract Features", "Measure Area"])
                    pipeline_placeholder.markdown(
                        render_pipeline_tracker(completed_steps),
                        unsafe_allow_html=True
                    )
                    st.success(f"✅ Normal Brain Scan — No Tumor Detected (Confidence: {det_conf * 100:.1f}%)")

                    if is_uncertain:
                        st.warning("⚠️ Low confidence / possible out-of-distribution scan. Radiologist review recommended.")

                    st.markdown("### 🔍 Image Acquisition & Preprocessing")
                    col_input, col_pre = st.columns(2)
                    with col_input:
                        st.subheader("Input MRI")
                        st.image(input_rgb, caption=f"Original Acquisition ({input_rgb.shape[1]}x{input_rgb.shape[0]})", width="stretch")
                    with col_pre:
                        st.subheader("Preprocessed")
                        st.image(preprocessed_rgb, caption="Standardized (256×256, Head-Cropped, uint8)", width="stretch")

                    st.divider()
                    st.markdown("### 📊 Diagnostic Evaluation")
                    m1, m2 = st.columns(2)
                    m1.metric("Screening Result", "Healthy / No Tumor")
                    m2.metric("Clearance Confidence", f"{det_conf * 100:.1f}%")

                    st.markdown("#### Deep Learning Class Probabilities (EfficientNet-B3)")
                    for cls_name, prob in probs.items():
                        percent = prob * 100
                        st.markdown(f"**{cls_name}**: `{percent:.1f}%`")
                        st.progress(min(max(float(prob), 0.0), 1.0))

                    st.info("ℹ️ Scan evaluation complete. No hyperintense mass lesion or pathological contrast enhancement identified.")

                    # Limitations expander
                    with st.expander("⚠️ Known Limitations", expanded=False):
                        st.markdown(_LIMITATIONS_TEXT)

                    st.stop()

                # --- Tumor detected path ---
                tumor_type = top_class
                class_conf = probs[top_class]
                completed_steps.append("Classify")

                # Step 4: Segment
                pipeline_placeholder.markdown(
                    render_pipeline_tracker(completed_steps, active_step="Segment"),
                    unsafe_allow_html=True
                )
                mask = cached_segment(preprocessed_rgb)
                seg_reliable = segmentation_ok(mask)
                completed_steps.append("Segment")

                # Step 5: Extract Features
                pipeline_placeholder.markdown(
                    render_pipeline_tracker(completed_steps, active_step="Extract Features"),
                    unsafe_allow_html=True
                )
                features = cached_extract_features(preprocessed_rgb, mask)
                completed_steps.append("Extract Features")

                # Step 6: Measure Area
                pipeline_placeholder.markdown(
                    render_pipeline_tracker(completed_steps, active_step="Measure Area"),
                    unsafe_allow_html=True
                )
                area_info = tumor_area(mask, px_spacing_mm=pixel_spacing)
                completed_steps.append("Measure Area")

                # Render all completed steps ticked
                pipeline_placeholder.markdown(
                    render_pipeline_tracker(completed_steps),
                    unsafe_allow_html=True
                )

                # Uncertainty banner
                if is_uncertain:
                    st.warning("⚠️ Low confidence / possible out-of-distribution scan. Radiologist review required.")
                
                # Detection Alert Banner
                st.error(f"⚠️ Tumor detected — {tumor_type} ({class_conf * 100:.1f}% conf.)")

                # 2. Acquisition & Preprocessed Columns
                st.markdown("### 🔍 Image Acquisition & Preprocessing")
                col_input, col_pre = st.columns(2)
                with col_input:
                    st.subheader("Input MRI")
                    st.image(
                        input_rgb,
                        caption=f"Original Acquisition ({input_rgb.shape[1]}x{input_rgb.shape[0]})",
                        width="stretch"
                    )

                with col_pre:
                    st.subheader("Preprocessed")
                    st.image(
                        preprocessed_rgb,
                        caption="Standardized (256×256, Head-Cropped, uint8)",
                        width="stretch"
                    )

                st.divider()

                # 3. Three Metric Cards
                st.markdown("### 📊 Diagnostic Evaluation")
                m1, m2, m3 = st.columns(3)
                m1.metric("Tumor Type", tumor_type)
                m2.metric("Classification Confidence", f"{class_conf * 100:.1f}%")

                # Area: honest labelling
                if seg_reliable:
                    m3.metric(
                        "Cross-Sectional Area (this slice)",
                        f"{area_info['area_value']:.2f} {area_info['unit']}"
                    )
                else:
                    m3.metric("Cross-Sectional Area (this slice)", "Segmentation unreliable")

                # Multi-Class Probability Distribution & Radiomic Features
                eval_col1, eval_col2 = st.columns([1, 1])

                with eval_col1:
                    st.markdown("#### Deep Learning Class Probabilities (EfficientNet-B3)")
                    for cls_name, prob in probs.items():
                        percent = prob * 100
                        st.markdown(f"**{cls_name}**: `{percent:.1f}%`")
                        st.progress(min(max(float(prob), 0.0), 1.0))

                with eval_col2:
                    st.markdown("#### Radiomic & Intensity Metrics")
                    if seg_reliable:
                        st.markdown(f"- **Mean Lesion Intensity:** `{features.get('mean_intensity', 0)}`")
                        st.markdown(f"- **Intensity Std Deviation:** `{features.get('std_intensity', 0)}`")
                        st.markdown(f"- **Maximum Intensity Peak:** `{features.get('max_intensity', 0)}`")
                        st.markdown(f"- **Cross-Sectional Coverage:** `{features.get('coverage_ratio', 0) * 100:.2f}%`")
                        st.markdown(f"- **Pixel Spacing Factor:** `{pixel_spacing:.2f} mm/px`")
                    else:
                        st.warning("Segmentation mask failed sanity check — radiomic metrics may be unreliable.")
                        st.markdown(f"- **Mean Lesion Intensity:** `{features.get('mean_intensity', 0)}`")
                        st.markdown(f"- **Pixel Spacing Factor:** `{pixel_spacing:.2f} mm/px`")

                # Precompute overlay and heatmap
                overlay_img = generate_tumor_overlay(preprocessed_rgb, mask)
                heatmap_img = cached_heatmap(preprocessed_rgb)

                st.divider()

                # 4. Visualization Tabs
                st.markdown("### 🔬 Visualization & Interpretability")

                tab_titles = ["Segmentation", "Overlay"]
                if show_heatmap:
                    tab_titles.append("Activation Heatmap (Surrogate)")

                tabs = st.tabs(tab_titles)

                with tabs[0]:
                    st.markdown("**Binary Segmentation Mask** (0 = Background, 1 = Tumor Tissue)")
                    st.image(
                        mask * 255,
                        caption="Segmented Tumor Region (Brain-Mask Constrained Thresholding)",
                        clamp=True,
                        width="stretch"
                    )
                    if not seg_reliable:
                        st.warning("⚠️ Segmentation mask failed sanity check (too small or too large). Results should be interpreted with caution.")

                with tabs[1]:
                    st.markdown("**Tumor Region Overlay** (Red Highlight on Preprocessed MRI)")
                    st.image(
                        overlay_img,
                        caption="Delineated Tumor Boundary and Spatial Extent",
                        width="stretch"
                    )

                if show_heatmap:
                    with tabs[2]:
                        st.markdown("**Segmentation-Centroid Heatmap (Surrogate)**")
                        st.caption("⚠️ This is NOT a true Grad-CAM. It is a Gaussian blob placed at the segmentation centroid and is not class-specific.")
                        st.image(
                            heatmap_img,
                            caption="Surrogate Activation Heatmap — NOT class-specific (JET Colormap)",
                            width="stretch"
                        )

                st.divider()

                # 5. Known Limitations (always shown)
                with st.expander("⚠️ Known Limitations", expanded=False):
                    st.markdown(_LIMITATIONS_TEXT)

                st.divider()

                # 6. Export Diagnostic Report Button
                st.markdown("### 📄 Clinical Export")
                report_bytes = generate_report_png(
                    input_img=input_rgb,
                    overlay_img=overlay_img,
                    heatmap_img=heatmap_img,
                    tumor_type=tumor_type,
                    confidence=class_conf,
                    area_info=area_info,
                    pixel_spacing=pixel_spacing,
                    filename=uploaded_file.name,
                    seg_reliable=seg_reliable,
                    uncertain=is_uncertain
                )

                st.download_button(
                    label="📥 Download Diagnostic Report (PNG)",
                    data=report_bytes,
                    file_name=f"tumor_diagnostic_report_{Path(uploaded_file.name).stem}.png",
                    mime="image/png",
                    use_container_width=True
                )

            except Exception as e:
                st.warning(f"⚠️ Error processing uploaded MRI scan: {e}")


# ============================================================================
# Shared limitations text (used in expander + report)
# ============================================================================
_LIMITATIONS_TEXT = """
- **Segmentation is heuristic** — intensity thresholding constrained within a brain mask, NOT a trained U-Net.
  Ring-enhancing lesions with a dark core will be partially missed.
- **Heatmap is a surrogate** — a Gaussian blob at the segmentation centroid, NOT a true Grad-CAM or Eigen-CAM.
  It is not class-specific. Real gradient-based explainability requires the original Keras model or a dual-output ONNX export.
- **Possible dataset shortcuts** — the training dataset (Kaggle brain tumor MRI) may contain patient-level data leakage
  and imaging shortcuts. Domain shift is expected on out-of-distribution scanners.
- **Area is from a single 2D slice** — not volumetric. Displayed in px² unless calibrated DICOM pixel spacing is provided.
- **Not a medical device** — this tool is a decision-support aid for academic purposes only. Final diagnosis must rest with
  a qualified radiologist/neurosurgeon.
"""
