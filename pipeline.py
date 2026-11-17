"""
pipeline.py - Backend processing and inference pipeline for Brain Tumor MRI Detection
ABES Engineering College, Dept. of CSE (Session 2026-27)

This module contains the core image processing, detection, segmentation,
classification, and explainability functions.

NOTE: Placeholder dummy implementations are currently active for UI testing.
Function signatures and return types are strictly defined to remain stable
when production deep learning models (e.g. PyTorch/TensorFlow) are integrated.
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


def predict_mri_probabilities(img: np.ndarray) -> Dict[str, float]:
    """
    Run inference through the deep learning model and return calibrated
    softmax probability distribution across all 4 clinical categories:
    [Glioma, Meningioma, No Tumor, Pituitary].
    """
    session = get_model_session()
    if session is not None:
        try:
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

            # Softmax safety check
            exp_scores = np.exp(raw_output - np.max(raw_output))
            probs = exp_scores / np.sum(exp_scores) if np.sum(raw_output) < 0.99 or np.sum(raw_output) > 1.01 else raw_output

            return {
                cls_name: round(float(prob), 4)
                for cls_name, prob in zip(_MODEL_CLASSES, probs)
            }
        except Exception as e:
            print(f"Error during ONNX inference: {e}")

    # Fallback deterministic pseudo-random distribution if model file is missing
    seed = int(np.mean(img) * 1000) % (2**31 - 1)
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.05, 0.95, size=4)
    probs = raw / np.sum(raw)
    return {cls_name: round(float(p), 4) for cls_name, p in zip(_MODEL_CLASSES, probs)}



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
    
    Args:
        image: PIL Image or numpy array.
        
    Returns:
        Dictionary containing dimensions, channels, mean, and std intensity.
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
# 2. Preprocessing
# ============================================================================

def preprocess(img: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Standardize MRI scan for model ingestion: resize to 256x256, apply Gaussian
    denoising filter, and ensure normalization to 8-bit unsigned integer (uint8).
    
    Args:
        img: Input image as numpy array (H, W, C or H, W) or PIL Image.
        
    Returns:
        Preprocessed image as np.ndarray with shape (256, 256, 3) and dtype uint8.
        
    TODO: Add skull stripping (HD-BET or Otsu-based masking) and bias field
          correction (N4ITK) when integrating clinical-grade MRI preprocessing.
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

    # 1. Resize to target dimension 256x256
    resized = cv2.resize(np_img, (256, 256), interpolation=cv2.INTER_AREA)

    # 2. Denoise with Gaussian Blur filter (kernel size 5x5)
    denoised = cv2.GaussianBlur(resized, (5, 5), sigmaX=0)

    # 3. Normalize intensity range to uint8 [0, 255]
    if denoised.dtype != np.uint8:
        norm = cv2.normalize(denoised, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        denoised = norm.astype(np.uint8)

    return denoised


# ============================================================================
# 3. Binary Tumor Detection
# ============================================================================

def detect(img: np.ndarray) -> Tuple[bool, float]:
    """
    Perform initial binary screening using the pre-trained EfficientNet-B3
    deep learning model to identify the presence of a brain tumor.
    
    Args:
        img: Preprocessed MRI image array of shape (256, 256, 3).
        
    Returns:
        Tuple of (tumor_present: bool, confidence: float).
    """
    probs = predict_mri_probabilities(img)
    no_tumor_prob = probs.get("No Tumor", 0.0)
    top_class = max(probs, key=probs.get)

    if top_class == "No Tumor":
        return False, round(float(no_tumor_prob), 4)
    else:
        # Tumor detected: confidence represents lesion presence probability
        tumor_conf = max(1.0 - no_tumor_prob, probs[top_class])
        return True, round(float(tumor_conf), 4)


# ============================================================================
# 4. Tumor Segmentation
# ============================================================================

def segment(img: np.ndarray) -> np.ndarray:
    """
    Segment the tumor region from the brain MRI scan, generating a binary mask.
    
    Args:
        img: Preprocessed MRI image array (256, 256, 3) or (256, 256).
        
    Returns:
        Binary mask as np.ndarray (256, 256) where 1 indicates tumor and 0 background.
        
    TODO: Replace morphological heuristic with deep learning segmentation model
          (e.g., U-Net, Attention U-Net, or TransUNet trained with Dice loss).
    """
    # Convert to grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Apply Otsu's thresholding to isolate hyperintense regions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological erosion followed by dilation (Opening) to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    dilated = cv2.dilate(opened, kernel, iterations=1)

    # Find contours and retain the largest significant contour
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape[:2], dtype=np.uint8)

    if contours:
        # Filter contours away from outer skull perimeter if possible
        h, w = gray.shape[:2]
        valid_contours = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            # Avoid the extreme border of the frame
            if x > 10 and y > 10 and (x + cw) < (w - 10) and (y + ch) < (h - 10):
                valid_contours.append(c)

        chosen = valid_contours if valid_contours else contours
        largest_contour = max(chosen, key=cv2.contourArea)
        # Draw binary mask filled with 1s
        cv2.drawContours(mask, [largest_contour], -1, 1, thickness=-1)
    else:
        # Fallback dummy elliptical mask near center if no contours detected
        cv2.circle(mask, (img.shape[1] // 2, img.shape[0] // 2), 25, 1, thickness=-1)

    return mask


# ============================================================================
# 5. Tumor Classification
# ============================================================================

def classify(img: np.ndarray) -> Tuple[str, float]:
    """
    Classify the identified brain tumor into its histopathological subtype
    (Glioma, Meningioma, Pituitary) using the fine-tuned EfficientNet-B3 model.
    
    Args:
        img: Preprocessed MRI image array (256, 256, 3).
        
    Returns:
        Tuple of (tumor_type: str, confidence: float).
    """
    probs = predict_mri_probabilities(img)
    tumor_probs = {k: v for k, v in probs.items() if k != "No Tumor"}
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
# 6. Explainable AI: Grad-CAM Overlay
# ============================================================================

def gradcam(img: np.ndarray) -> np.ndarray:
    """
    Generate visual explainability heatmap overlay (Grad-CAM) highlighting
    regions salient to the classification decision.
    
    Args:
        img: Preprocessed MRI image array (256, 256, 3) in uint8 format.
        
    Returns:
        RGB image with JET colormap heatmap blended over the MRI scan.
        
    TODO: Replace synthetic Gaussian blob with actual gradient-weighted activation
          map computed from the final convolutional layer of the classification model.
    """
    # Ensure RGB base
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
    
    # Normalize to 0-255 uint8
    heatmap_gray = np.clip(gaussian * 255, 0, 255).astype(np.uint8)

    # Apply JET color map
    heatmap_jet = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

    # Alpha-blend the heatmap with the original MRI image (60% base, 40% heatmap)
    blended = cv2.addWeighted(base_img, 0.65, heatmap_rgb, 0.35, 0)

    return blended


# ============================================================================
# 7. Quantitative Metrics: Tumor Area Calculation
# ============================================================================

def tumor_area(mask: np.ndarray, px_spacing_mm: float = 1.0) -> float:
    """
    Calculate the cross-sectional 2D tumor area in square millimeters (mm²).
    
    Formula:
        Area (mm²) = (Total Positive Mask Pixels) * (Pixel Spacing in mm)²
        
    Args:
        mask: Binary segmentation mask array with values in {0, 1}.
        px_spacing_mm: In-plane voxel/pixel spacing in millimeters (default: 1.0 mm).
        
    Returns:
        Estimated tumor cross-sectional surface area in mm².
    """
    pixel_count = int(np.sum(mask > 0))
    area_mm2 = float(pixel_count * (px_spacing_mm ** 2))
    return round(area_mm2, 2)


# ============================================================================
# 8. Radiomic Feature Extraction
# ============================================================================

def extract_features(img: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Extract first-order statistical and morphological features from the
    segmented tumor region for clinical characterization.
    
    Args:
        img: Preprocessed MRI image array (256, 256, 3).
        mask: Binary tumor mask (256, 256) where tumor = 1.
        
    Returns:
        Dictionary containing statistical metrics (mean, std, max, solidity).
        
    TODO: Integrate PyRadiomics for deep radiomic extraction (GLCM, GLRLM, shape).
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

