import os
import sys
import json
import uuid
import torch
import numpy as np
import urllib.request
from PIL import Image
import cv2

# Import SAM specific tools
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ============================================================
# SAM Weights Auto-Downloader
# ============================================================
SAM_MODELS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}

def get_weights(model_type, weights_dir):
    os.makedirs(weights_dir, exist_ok=True)
    filename = os.path.basename(SAM_MODELS[model_type])
    local_path = os.path.join(weights_dir, filename)
    
    if not os.path.exists(local_path):
        print(f"[SAM] Downloading {model_type} weights... this is a one-time thing.")
        urllib.request.urlretrieve(SAM_MODELS[model_type], local_path)
    return local_path

# ============================================================
# SAM Adapter (Inference)
# ============================================================

def run(
    image_path: str,
    out_root: str = ".",
    model_type: str = "vit_b",  # UI selects vit_b, vit_l, or vit_h
    points_per_side: int = 32,
    **params
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Resolve Weights
    weights_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "weights"))
    checkpoint_path = get_weights(model_type, weights_dir)

    # 2. Initialize SAM Model
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
    )

    # 3. Load Image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # 4. Predict Masks
    print(f"[SAM] Processing {os.path.basename(image_path)} on {device}...")
    masks = mask_generator.generate(img_np)

    if not masks:
        raise ValueError("SAM failed to generate any masks for this image.")

    # 5. Post-Process (Combine all masks into one visual)
    # SAM returns a list of dictionaries. We'll merge them for the preview.
    full_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
    for m in masks:
        full_mask[m['segmentation']] = 255

    # Create segmented visual (Image * Mask)
    mask_bool = full_mask > 0
    segmented = img_np.copy()
    segmented[~mask_bool] = 0 # Black out non-segmented areas

    # --------------------------------------------------------
    # JSON payload
    # --------------------------------------------------------
    payload = {
        "tool": "SAM_Segmentation",
        "tool_version": {"torch": torch.__version__, "python": sys.version.split()[0]},
        "image": {
            "original_path": image_path,
            "objects_found": len(masks),
            "original_shape": list(img_np.shape),
        },
        "sam_parameters_used": {
            "model_type": model_type,
            "points_per_side": points_per_side
        }
    }

    # --------------------------------------------------------
    # Save outputs (Mirroring your U-Net structure)
    # --------------------------------------------------------
    base = os.path.splitext(os.path.basename(image_path))[0]
    uid = uuid.uuid4().hex[:8]
    stem = f"{base}_sam_{uid}"

    out_root_abs = os.path.abspath(out_root or ".")
    algo_dir = os.path.join(out_root_abs, "features", "sam_segmentation")
    os.makedirs(algo_dir, exist_ok=True)

    json_path = os.path.join(algo_dir, stem + ".json")
    mask_path = os.path.join(algo_dir, stem + "_mask.png")
    vis_path = os.path.join(algo_dir, stem + "_segmented.jpg")

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=4)

    Image.fromarray(full_mask).save(mask_path)
    Image.fromarray(segmented).save(vis_path)

    return os.path.abspath(json_path), os.path.abspath(mask_path), os.path.abspath(vis_path)

if __name__ == "__main__":
    # Test call
    j, m, v = run(
        image_path="sample.png", # Ensure this file exists for testing
        out_root="./outputs",
        model_type="vit_b",
        points_per_side=32
    )
    print(f"Success! Results in: {os.path.dirname(j)}")