"""
pipeline.py - Backend processing and inference pipeline for Brain Tumor MRI Detection
ABES Engineering College, Dept. of CSE (Session 2026-27)

This module contains the core image processing, detection, segmentation,
classification, and explainability functions.

Improvements applied:
  A. Single inference via analyze() + uncertainty/reject + temperature scaling
  B. Head crop preprocessing (crop_to_head) — Gaussian blur removed
  C. Brain-mask-constrained segmentation + segmentation_ok sanity check
  D. Honest area reporting (px² when uncalibrated)
  E. Honest heatmap labeling (surrogate, not Grad-CAM)
"""

from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# ============================================================================
# Deep Learning Model (ONNX Runtime - EfficientNet-B3)
# ============================================================================
_ONNX_SESSION = None
_MODEL_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# --- Calibration / Uncertainty knobs ---
TEMPERATURE = 1.0          # tune on a validation set (NLL-minimise)
CONF_THRESHOLD = 0.80      # top-class probability below this ⇒ uncertain
ENTROPY_MAX = 0.90          # entropy above this ⇒ uncertain


def get_model_session():
    """
    Load or retrieve cached ONNX inference session for the pre-trained
    EfficientNet-B3 Brain Tumor Classification model.
    """
    global _ONNX_SESSION
    if _ONNX_SESSION is None:
        try:
            import onnxruntime as ort
            model_path = Path(__file__).parent / "models" / "efficientnetb3.onnx"
            if model_path.exists():
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                _ONNX_SESSION = ort.InferenceSession(
                    str(model_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
        except Exception as e:
            print(f"Notice: ONNX Runtime initialization fallback: {e}")
            _ONNX_SESSION = None
    return _ONNX_SESSION


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax (no scipy dependency)."""
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / e.sum()


def predict_mri_probabilities(img: np.ndarray) -> Dict[str, float]:
    """
    Run inference through the deep learning model and return calibrated
    softmax probability distribution across all 4 clinical categories:
    [Glioma, Meningioma, No Tumor, Pituitary].

    Raises RuntimeError if model file is missing.
    """
    session = get_model_session()
    if session is None:
        raise RuntimeError(
            "Model file not found: models/efficientnetb3.onnx — "
            "download it from HuggingFace (Udit013/brain-tumor-efficientnetb3)."
        )

    # Ensure standard (256, 256, 3) float32 in range [0, 255]
    if img.shape[:2] != (256, 256):
        processed = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    else:
        processed = img.copy()

    if processed.ndim == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    elif processed.ndim == 3 and processed.shape[2] == 4:
        processed = cv2.cvtColor(processed, cv2.COLOR_RGBA2RGB)

    input_tensor = processed.astype(np.float32)[np.newaxis, ...]
    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: input_tensor})[0][0]

    # Temperature-scaled re-calibration:
    # Model outputs softmax probs → convert to log-space → scale → re-softmax
    logits = np.log(np.clip(raw_output, 1e-8, 1.0)) / TEMPERATURE
    probs = _softmax(logits)

    return {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(_MODEL_CLASSES, probs)
    }


# ============================================================================
# A. Single-inference wrapper with uncertainty rejection
# ============================================================================

def analyze(img: np.ndarray) -> Dict[str, Any]:
    """
    Run the model **once** and return a rich result dict:
      - probs:      {class_name: float}
      - top:        str (highest-probability class)
      - entropy:    float (Shannon entropy of the distribution)
      - uncertain:  bool (True when confidence is too low or entropy too high)
      - tumor_prob: float (1 - P(No Tumor))
    """
    probs = predict_mri_probabilities(img)
    p = np.array(list(probs.values()))
    entropy = float(-(p * np.log(p + 1e-8)).sum())
    top = max(probs, key=probs.get)
    uncertain = probs[top] < CONF_THRESHOLD or entropy > ENTROPY_MAX

    return {
        "probs": probs,
        "top": top,
        "entropy": entropy,
        "uncertain": uncertain,
        "tumor_prob": 1.0 - probs["No Tumor"],
    }


# ============================================================================
# 1. Image Loading & Metadata Utilities (Helper Functions)
# ============================================================================

def load_image(image_file: Any) -> Optional[Image.Image]:
    """
    Load an uploaded image into a standard RGB PIL Image object.

    Args:
        image_file: File-like object (BytesIO or Streamlit UploadedFile) or file path.

    Returns:
        PIL.Image.Image in RGB mode, or None if reading fails.
    """
    try:
        image = Image.open(image_file).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def get_image_metadata(image: Union[Image.Image, np.ndarray]) -> Dict[str, Any]:
    """
    Extract dimensional and intensity metadata from an MRI scan.
    """
    if isinstance(image, Image.Image):
        np_img = np.array(image)
        width, height = image.width, image.height
    else:
        np_img = image
        height, width = np_img.shape[:2]

    return {
        "dimensions": f"{width} x {height}",
        "channels": np_img.shape[2] if np_img.ndim == 3 else 1,
        "mean_intensity": round(float(np.mean(np_img)), 2),
        "std_intensity": round(float(np.std(np_img)), 2)
    }


# ============================================================================
# B. Head-crop utility (Kaggle-style input matching)
# ============================================================================

def crop_to_head(rgb: np.ndarray, margin: int = 8) -> np.ndarray:
    """
    Crop the image tightly around the brain/head region, discarding excess
    black background.  This replicates the tight-crop commonly seen in
    Kaggle brain-tumor datasets.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return rgb
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    H, W = rgb.shape[:2]
    return rgb[max(y - margin, 0):min(y + h + margin, H),
               max(x - margin, 0):min(x + w + margin, W)]


# ============================================================================
# 2. Preprocessing
# ============================================================================

def preprocess(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Standardize MRI scan for model ingestion:
      1. Convert to RGB numpy array
      2. Crop tightly around head (crop_to_head)
      3. Resize to 256×256
      4. Normalize intensity to uint8 [0, 255]

    NOTE: Gaussian blur removed — it was not part of the training pipeline
          and hurts fine detail.
    """
    # Convert PIL Image to numpy array if necessary
    if isinstance(img, Image.Image):
        np_img = np.array(img)
    else:
        np_img = img.copy()

    # Ensure 3-channel RGB representation
    if np_img.ndim == 2:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
    elif np_img.ndim == 3 and np_img.shape[2] == 4:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGBA2RGB)

    # 1. Tight crop around head/brain
    np_img = crop_to_head(np_img)

    # 2. Resize to target dimension 256x256
    resized = cv2.resize(np_img, (256, 256), interpolation=cv2.INTER_AREA)

    # 3. Normalize intensity range to uint8 [0, 255]
    if resized.dtype != np.uint8:
        norm = cv2.normalize(resized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        resized = norm.astype(np.uint8)

    return resized


# ============================================================================
# 3. Binary Tumor Detection (now wraps analyze())
# ============================================================================

def detect(img: np.ndarray) -> Tuple[bool, float]:
    """
    Perform initial binary screening using the pre-trained EfficientNet-B3
    deep learning model to identify the presence of a brain tumor.

    Returns:
        Tuple of (tumor_present: bool, confidence: float).
    """
    result = analyze(img)
    top = result["top"]
    probs = result["probs"]

    if top == "No Tumor":
        return False, round(float(probs["No Tumor"]), 4)
    else:
        tumor_conf = max(result["tumor_prob"], probs[top])
        return True, round(float(tumor_conf), 4)


# ============================================================================
# C. Tumor Segmentation — brain-mask-constrained + sanity check
# ============================================================================

def brain_mask(gray: np.ndarray) -> np.ndarray:
    """
    Compute a binary mask of the brain parenchyma, excluding skull and
    scalp. Uses adaptive background thresholding + morphological closing + erosion.
    """
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # If the image was tightly cropped to head, Otsu overshoots because background is minimal.
    # Fallback to fixed threshold (20) to separate true MRI air background from brain tissue.
    _, th_bg = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    th = th_bg if np.count_nonzero(th_otsu) < 0.25 * gray.size else th_otsu

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m = np.zeros_like(gray)
    if cnts:
        cv2.drawContours(m, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return cv2.erode(m, k, iterations=1)   # remove skull/scalp ring


def segment(img: np.ndarray, k_sigma: float = 1.2) -> np.ndarray:
    """
    Segment the tumor region from the brain MRI scan using intensity
    thresholding constrained within the brain mask.

    Returns:
        Binary mask (H, W) where 1 = tumor, 0 = background.
        Returns all zeros when no valid bright region is found.

    NOTE: This is still a heuristic approach. Ring-enhancing lesions with a
          dark core will be partially missed.  For production use, train a
          U-Net on the Figshare (Cheng) dataset which provides ground-truth
          tumor masks.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img.copy()
    bm = brain_mask(gray) > 0
    vals = gray[bm]
    if vals.size == 0:
        return np.zeros(gray.shape, np.uint8)

    thr = vals.mean() + k_sigma * vals.std()
    m = ((gray > thr) & bm).astype(np.uint8)
    m = cv2.morphologyEx(
        m, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(gray.shape, np.uint8)

    best = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lab == best).astype(np.uint8)


def segmentation_ok(mask: np.ndarray, min_px: int = 30, max_cov: float = 0.20) -> bool:
    """
    Sanity-check the segmentation mask.
    Returns False if the mask is too small (<min_px pixels) or covers
    more than max_cov fraction of the image (likely a skull artefact).
    """
    total = mask.sum()
    cov = total / mask.size
    return total >= min_px and cov <= max_cov


# ============================================================================
# 5. Tumor Classification (wraps analyze() — single inference)
# ============================================================================

def classify(img: np.ndarray) -> Tuple[str, float]:
    """
    Classify the identified brain tumor into its histopathological subtype
    (Glioma, Meningioma, Pituitary) using the fine-tuned EfficientNet-B3 model.

    Returns:
        Tuple of (tumor_type: str, confidence: float).
    """
    result = analyze(img)
    tumor_probs = {k: v for k, v in result["probs"].items() if k != "No Tumor"}
    if tumor_probs:
        best_type = max(tumor_probs, key=tumor_probs.get)
        confidence = tumor_probs[best_type]
        return best_type, round(float(confidence), 4)
    return "Glioma", 0.85


def get_prediction_breakdown(img: np.ndarray) -> Dict[str, float]:
    """
    Returns full probability distribution across all 4 clinical classes for UI display.
    """
    return predict_mri_probabilities(img)


# ============================================================================
# E. Explainability: Surrogate Heatmap (honestly labeled — NOT Grad-CAM)
# ============================================================================

def surrogate_heatmap(img: np.ndarray) -> np.ndarray:
    """
    Generate a **surrogate** visual heatmap overlay centred on the
    segmentation mask centroid.

    ⚠️  This is NOT a true Grad-CAM or Eigen-CAM.  It is a Gaussian blob
    placed at the segmentation centroid and is therefore *not* class-specific.
    Real gradient-based explainability requires either:
      - The original Keras model with tf.GradientTape (Grad-CAM), or
      - A dual-output ONNX export (feature map + logits) for Eigen-CAM.

    Returns:
        RGB image (H, W, 3) with JET heatmap blended over the MRI.
    """
    if img.ndim == 2:
        base_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        base_img = img.copy()

    h, w = base_img.shape[:2]

    # Use segmentation mask centroid for focused Gaussian activation
    mask = segment(base_img)
    moments = cv2.moments(mask.astype(np.uint8))

    if moments["m00"] > 0:
        center_x = int(moments["m10"] / moments["m00"])
        center_y = int(moments["m01"] / moments["m00"])
    else:
        center_x, center_y = w // 2, h // 2

    # Synthesize 2D Gaussian heatmap centered on tumor location
    y_coords, x_coords = np.ogrid[:h, :w]
    sigma = min(h, w) / 7.0
    gaussian = np.exp(-((x_coords - center_x)**2 + (y_coords - center_y)**2) / (2 * (sigma**2)))

    heatmap_gray = np.clip(gaussian * 255, 0, 255).astype(np.uint8)
    heatmap_jet = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(base_img, 0.65, heatmap_rgb, 0.35, 0)
    return blended


# Keep old name as alias for backward-compat (app.py imports)
gradcam = surrogate_heatmap


# ============================================================================
# D. Quantitative Metrics: Tumor Area Calculation (honest units)
# ============================================================================

def tumor_area(mask: np.ndarray, px_spacing_mm: float = 1.0) -> Dict[str, Any]:
    """
    Calculate the cross-sectional 2D tumor area.

    Returns a dict:
      - pixel_count:  int — raw number of positive pixels
      - area_value:   float — numerical area
      - unit:         str — "mm²" if calibrated, "px² (uncalibrated)" otherwise
      - calibrated:   bool
    """
    pixel_count = int(np.sum(mask > 0))
    calibrated = abs(px_spacing_mm - 1.0) > 1e-6  # True when user set a real spacing
    area_value = float(pixel_count * (px_spacing_mm ** 2))

    return {
        "pixel_count": pixel_count,
        "area_value": round(area_value, 2),
        "unit": "mm²" if calibrated else "px² (uncalibrated)",
        "calibrated": calibrated,
    }


# ============================================================================
# 8. Radiomic Feature Extraction
# ============================================================================

def extract_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Extract first-order statistical and morphological features from the
    segmented tumor region for clinical characterization.
    """
    tumor_pixels = img[mask > 0] if np.any(mask > 0) else img
    h, w = mask.shape[:2]
    total_area = float(h * w)
    tumor_pixels_count = float(np.sum(mask > 0))

    return {
        "mean_intensity": round(float(np.mean(tumor_pixels)), 2),
        "std_intensity": round(float(np.std(tumor_pixels)), 2),
        "max_intensity": int(np.max(tumor_pixels)),
        "coverage_ratio": round(tumor_pixels_count / max(total_area, 1.0), 4)
    }
