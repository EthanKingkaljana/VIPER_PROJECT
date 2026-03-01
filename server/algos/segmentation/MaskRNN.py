# server/algos/segmentation/MaskRNN.py

import os
import sys
import json
import uuid
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import ToTensor
import torchvision


# ============================================================
# COCO class names
# ============================================================

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# ============================================================
# Load model
# ============================================================

def load_model(device):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model


# ============================================================
# Main Runner
# ============================================================

def run(image_path: str,
        out_root: str = ".",
        model_path: str = None,
        score_thr: float = 0.5,
        **params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    # ----------------------------
    # Load Image
    # ----------------------------
    img = Image.open(image_path).convert("RGB")
    img_tensor = ToTensor()(img).to(device)

    with torch.no_grad():
        pred = model([img_tensor])[0]

    masks = pred["masks"][:, 0]
    labels = pred["labels"]
    scores = pred["scores"]
    boxes = pred["boxes"]

    # Filter by score threshold
# Filter by score
    keep = scores > score_thr
    boxes = boxes[keep]
    masks = masks[keep]
    labels = labels[keep]
    scores = scores[keep]

    # Apply Non-Maximum Suppression
    nms_indices = torchvision.ops.nms(boxes, scores, iou_threshold=0.4)

    boxes = boxes[nms_indices]
    masks = masks[nms_indices]
    labels = labels[nms_indices]
    scores = scores[nms_indices]

    H, W = img_tensor.shape[1:]

    # ----------------------------
    # Combine masks
    # ----------------------------
    if len(masks) == 0:
        combined_mask = torch.zeros((H, W), device=device)
    else:
        combined_mask = (masks > 0.5).float().max(dim=0)[0]

    mask_np = combined_mask.cpu().numpy()

    # ----------------------------
    # Create segmentation overlay
    # ----------------------------
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np_uint8 = (img_np * 255).astype(np.uint8)

    overlay = img_np.copy()
    overlay[mask_np == 0] = 0
    overlay = (overlay * 255).astype(np.uint8)

    mask_img = (mask_np * 255).astype(np.uint8)

    # ----------------------------
    # Detection visualization (Bounding Boxes)
    # ----------------------------
    detection_vis = img_np_uint8.copy()

    boxes_np = boxes.cpu().numpy()
    labels_np = labels.cpu().numpy()
    scores_np = scores.cpu().numpy()

    for i in range(len(boxes_np)):
        x1, y1, x2, y2 = boxes_np[i].astype(int)
        class_id = int(labels_np[i])
        class_name = COCO_CLASSES[class_id]
        confidence = scores_np[i]

        # Draw red rectangle
        cv2.rectangle(detection_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Label text
        text = f"{class_name}: {confidence:.2f}"
        cv2.putText(
            detection_vis,
            text,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA
        )

    # ----------------------------
    # Build detections list
    # ----------------------------
    detections = []
    for i in range(len(labels_np)):
        detections.append({
            "class_id": int(labels_np[i]),
            "class_name": COCO_CLASSES[int(labels_np[i])],
            "confidence": float(scores_np[i])
        })

    # ----------------------------
    # JSON payload
    # ----------------------------
    payload = {
        "tool": "MaskRCNN_Segmentation",
        "tool_version": {
            "torch": torch.__version__,
            "python": sys.version.split()[0]
        },
        "detections": detections,
        "image": {
            "original_path": image_path,
            "file_name": os.path.basename(image_path),
            "original_shape": list(img.size[::-1]),
            "mask_shape": list(mask_img.shape),
            "segmented_shape": list(overlay.shape)
        },
        "maskrcnn_parameters_used": {
            "score_thr": score_thr,
            "model_path": model_path
        }
    }

    # ----------------------------
    # Save outputs
    # ----------------------------
    base = os.path.splitext(os.path.basename(image_path))[0]
    uid = uuid.uuid4().hex[:8]
    stem = f"{base}_maskrcnn_{uid}"

    out_root_abs = os.path.abspath(out_root or ".")
    algo_dir = os.path.join(out_root_abs, "features", "maskrcnn_segmentation")
    os.makedirs(algo_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(algo_dir, stem + ".json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=4)

    # Mask image
    mask_path = os.path.join(algo_dir, stem + "_mask.png")
    Image.fromarray(mask_img).save(mask_path)

    # Segmented overlay
    overlay_path = os.path.join(algo_dir, stem + "_segmented.jpg")
    Image.fromarray(overlay).save(overlay_path)

    # Detection visualization
    detection_path = os.path.join(algo_dir, stem + "_detection.jpg")
    Image.fromarray(detection_vis).save(detection_path)

    return (
        os.path.abspath(json_path),
        os.path.abspath(mask_path),
        os.path.abspath(overlay_path),
        os.path.abspath(detection_path)
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    j, m, s, d = run("your_image.jpg", "./maskrcnn_output")
    print("JSON:", j)
    print("Mask:", m)
    print("Segmented:", s)
    print("Detection:", d)