import os
import json
import uuid
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

# ===============================
# Global Model Cache
# ===============================
_MODEL_CACHE = {}

VOC_CLASSES = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"
]

# ===============================
# VOC Official Colormap
# ===============================
def voc_colormap():
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        cmap[i] = np.array([r, g, b])
    return cmap


# ===============================
# Model Loader
# ===============================
def get_model(device, model_path=None):
    if "model" not in _MODEL_CACHE:
        weights = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT
        model = models.segmentation.deeplabv3_resnet101(weights=weights)

        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))

        model.to(device).eval()
        _MODEL_CACHE["model"] = model
    return _MODEL_CACHE["model"]


# ===============================
# Main Run Function
# ===============================
def run(image_path: str,
        out_root: str=".",
        model_path: str=None,
        **params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device, model_path)

    # ---------------------------
    # 1. Load Image
    # ---------------------------
    input_image = Image.open(image_path).convert("RGB")
    img_np = np.array(input_image)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image).unsqueeze(0).to(device)

    # ---------------------------
    # 2. Inference
    # ---------------------------
    with torch.no_grad():
        output = model(input_tensor)["out"]

        # Upsample to original image size
        output = torch.nn.functional.interpolate(
            output,
            size=img_np.shape[:2],
            mode="bilinear",
            align_corners=False
        )[0]

        probs = torch.nn.functional.softmax(output, dim=0)
        conf, pred = torch.max(probs, dim=0)

    pred = pred.cpu().numpy()
    conf = conf.cpu().numpy()

    # ---------------------------
    # 3. Confidence Filtering
    # ---------------------------
    threshold = params.get("threshold", 0.0)
    pred[conf < threshold] = 0

    # ---------------------------
    # 4. Colorize Segmentation
    # ---------------------------
    cmap = voc_colormap()
    color_mask = cmap[pred]  # (H, W, 3)

    # Overlay on original image
    alpha = 0.6
    overlay = (img_np * (1 - alpha) + color_mask * alpha).astype(np.uint8)

    # ---------------------------
    # 5. Class Detection
    # ---------------------------
    class_ids = np.unique(pred)
    classes = [VOC_CLASSES[c] for c in class_ids if c < len(VOC_CLASSES)]

    # ---------------------------
    # 6. Save Outputs
    # ---------------------------
    base = os.path.splitext(os.path.basename(image_path))[0]
    uid = uuid.uuid4().hex[:8]
    stem = f"{base}_deeplab_{uid}"

    algo_dir = os.path.join(os.path.abspath(out_root),
                            "features",
                            "deeplabv3_outputs")
    os.makedirs(algo_dir, exist_ok=True)

    json_path = os.path.join(algo_dir, stem + ".json")
    mask_path = os.path.join(algo_dir, stem + "_mask.png")
    vis_path = os.path.join(algo_dir, stem + "_overlay.jpg")

    with open(json_path, "w") as f:
        json.dump({
            "detected_classes": classes,
            "confidence_threshold": threshold
        }, f, indent=4)

    Image.fromarray(color_mask).save(mask_path)
    Image.fromarray(overlay).save(vis_path)

    return json_path, mask_path, vis_path