import os
import sys
import json
import uuid
import cv2
import numpy as np

def run(image_path: str, out_root: str = ".", **params):
    # -------------------------------
    # Load image
    # -------------------------------
    # We load with IMREAD_UNCHANGED to detect if there's an alpha channel
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # -------------------------------
    # AUTO-CONVERT TO GRAYSCALE
    # -------------------------------
    # If the image has 3 (BGR) or 4 (BGRA) channels, convert it.
    if img.ndim > 2:
        if img.shape[2] == 4:
            # Drop alpha channel first
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Convert BGR to Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # At this point, img is guaranteed to be 2D/grayscale.

    # -------------------------------
    # CLAHE PARAMETERS
    # -------------------------------
    clipLimit = float(params.get("clipLimit", 3.0))
    tileGridSize = params.get("tileGridSize", (8, 8))
    if isinstance(tileGridSize, str):
        # allow passing as string like "8,8"
        tileGridSize = tuple(map(int, tileGridSize.split(",")))

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    
    # -------------------------------
    # Apply CLAHE
    # -------------------------------
    enhanced = clahe.apply(img)

    # -------------------------------
    # Prepare payload
    # -------------------------------
    payload = {
        "tool": "CLAHE",
        "tool_version": {"opencv": cv2.__version__, "python": sys.version.split()[0]},
        "image": {
            "original_path": image_path,
            "file_name": os.path.basename(image_path),
            "original_shape": list(img.shape), # This will now reflect the grayscale shape
            "enhanced_shape": list(enhanced.shape),
            "dtype": str(enhanced.dtype)
        },
        "clahe_parameters_used": {
            "clipLimit": clipLimit,
            "tileGridSize": tileGridSize
        }
    }

    # -------------------------------
    # Unique stem
    # -------------------------------
    base = os.path.splitext(os.path.basename(image_path))[0]
    unique_id = uuid.uuid4().hex[:8]
    stem = f"{base}_clahe_{unique_id}"

    # -------------------------------
    # Output directories
    # -------------------------------
    out_root_abs = os.path.abspath(out_root or ".")
    algo_dir = os.path.join(out_root_abs, "features", "clahe_outputs")
    os.makedirs(algo_dir, exist_ok=True)

    # -------------------------------
    # Save JSON
    # -------------------------------
    json_path = os.path.join(algo_dir, stem + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    json_path = os.path.abspath(json_path)

    # -------------------------------
    # Save Visualization
    # -------------------------------
    vis_path = os.path.join(algo_dir, stem + "_vis.jpg")
    cv2.imwrite(vis_path, enhanced)
    vis_path = os.path.abspath(vis_path)

    return json_path, vis_path

if __name__ == "__main__":
    # Test with any image type now!
    img_path = "test_image.jpg" 
    out_dir = "./clahe_output"
    json_file, vis_file = run(img_path, out_dir, clipLimit=2.0, tileGridSize=(8,8))
    print("JSON saved at:", json_file)
    print("Enhanced image saved at:", vis_file)