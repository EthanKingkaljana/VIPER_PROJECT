import os
import sys
import json
import uuid
import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Tuple

# ============================================================
# 1. Zero-DCE Model
# ============================================================

class ZeroDCE(nn.Module):
    def __init__(self):
        super().__init__()
        nf = 32
        self.relu = nn.ReLU(inplace=True)

        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1)

        self.e_conv5 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv6 = nn.Conv2d(nf * 2, nf, 3, 1, 1)
        self.e_conv7 = nn.Conv2d(nf * 2, 24, 3, 1, 1)

    def forward(self, x, num_iters=8):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        curves = torch.split(x_r, 3, dim=1)

        limit = min(len(curves), max(1, int(num_iters)))

        for i in range(limit):
            x = x + curves[i] * (x * x - x)
            x = torch.clamp(x, 0, 1)

        return x


# ============================================================
# 2. Global Model Cache
# ============================================================

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None
_MODEL_PATH = None


def get_default_model_path():
    """
    Resolves:
    server/weights/ZeroDCE_epoch99.pth
    from:
    server/algos/enhancement/Zero.py
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(
        os.path.join(base_dir, "..", "..", "weights", "ZeroDCE_epoch99.pth")
    )



def load_model(model_path: str):
    global _MODEL, _MODEL_PATH

    model_path = os.path.abspath(model_path)

    if _MODEL is not None and _MODEL_PATH == model_path:
        return _MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[Zero-DCE] Weights not found at:\n{model_path}"
        )

    model = ZeroDCE().to(_DEVICE)

    state_dict = torch.load(model_path, map_location=_DEVICE)

    # Remove DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    if _DEVICE.type == "cuda":
        model.half()

    _MODEL = model
    _MODEL_PATH = model_path
    return model


# ============================================================
# 3. Inference
# ============================================================

def run(
    image_path: str,
    out_root: str = ".",
    model_path: str = None,
    iterations: int = 8
) -> Tuple[str, str]:

    if model_path is None:
        model_path = get_default_model_path()

    model = load_model(model_path)

    # ---------------------------
    # Load Image
    # ---------------------------
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(_DEVICE)

    if _DEVICE.type == "cuda":
        img_tensor = img_tensor.half()

    # ---------------------------
    # Inference (optimized)
    # ---------------------------
    with torch.inference_mode():
        enhanced_tensor = model(img_tensor, num_iters=iterations)

    enhanced = enhanced_tensor.squeeze(0).float().cpu().numpy()
    enhanced = np.transpose(enhanced, (1, 2, 0))
    enhanced = np.clip(enhanced * 255.0, 0, 255).astype(np.uint8)

    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)

    # ============================================================
    # Save Output
    # ============================================================

    base = os.path.splitext(os.path.basename(image_path))[0]
    unique_id = uuid.uuid4().hex[:8]
    stem = f"{base}_zero_dce_{unique_id}"

    algo_dir = os.path.join(os.path.abspath(out_root), "features", "zero_dce_outputs")
    os.makedirs(algo_dir, exist_ok=True)

    json_path = os.path.join(algo_dir, stem + ".json")
    vis_path = os.path.join(algo_dir, stem + "_vis.jpg")

    payload = {
        "tool": "Zero-DCE",
        "device": str(_DEVICE),
        "tool_version": {
            "torch": torch.__version__,
            "python": sys.version.split()[0]
        },
        "image": {
            "original_path": image_path,
            "original_shape": list(img_bgr.shape),
            "enhanced_shape": list(enhanced.shape),
        },
        "parameters": {
            "iterations": int(iterations),
            "model": os.path.basename(model_path)
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    cv2.imwrite(vis_path, enhanced_bgr)

    return os.path.abspath(json_path), os.path.abspath(vis_path)


# ============================================================
# 4. CLI Test
# ============================================================

if __name__ == "__main__":
    try:
        j, v = run("test.jpg", iterations=8)
        print("Saved JSON:", j)
        print("Saved Image:", v)
    except Exception as e:
        print("[Zero-DCE] 💥 Error:", e)
