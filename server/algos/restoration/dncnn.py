import os
import sys
import json
import uuid
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor


# ==========================
# DnCNN Model Definition
# ==========================
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()

        kernel_size = 3
        padding = 1
        features = 64

        layers = []

        # First layer
        layers.append(
            nn.Conv2d(channels, features, kernel_size, padding=padding, bias=False)
        )
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(features, features, kernel_size, padding=padding, bias=False)
            )
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        layers.append(
            nn.Conv2d(features, channels, kernel_size, padding=padding, bias=False)
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # Residual learning
        noise = self.dncnn(x)
        return x - noise


# ==========================
# Main Run Function
# ==========================
def run(image_path: str, out_root: str = ".", model_path: str = None, **params):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Sigma parameter
    sigma = int(params.get("sigma", 25))

    # ------------------------------
    # Locate weights folder
    # ------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_dir = os.path.abspath(os.path.join(current_dir, "../../"))
    weights_dir = os.path.join(server_dir, "weights")

    # Select model based on sigma
    if model_path is None:
        if sigma <= 15:
            model_name = "dncnn_15.pth"
        elif sigma <= 25:
            model_name = "dncnn_25.pth"
        else:
            model_name = "dncnn_50.pth"

        model_path = os.path.join(weights_dir, model_name)

    # ------------------------------
    # Initialize Model
    # ------------------------------
    model = DnCNN(channels=1).to(device)

    # ------------------------------
    # Load Weights (FIXED)
    # ------------------------------
    if os.path.exists(model_path):

        state_dict = torch.load(model_path, map_location=device)

        # If saved with DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        print(f"[DnCNN] Loaded weights from {model_path}")

    else:
        print(f"[DnCNN] ⚠ Weights not found at {model_path}")
        print("[DnCNN] Running with randomly initialized weights")

    model.eval()

    # ------------------------------
    # Load Image (Grayscale for X-ray)
    # ------------------------------
    img = Image.open(image_path).convert("L")  # grayscale
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    # ------------------------------
    # Denoising
    # ------------------------------
    with torch.no_grad():
        out_tensor = model(img_tensor)

    out_img = out_tensor.squeeze().cpu().clamp(0, 1).numpy()
    out_img = (out_img * 255).astype(np.uint8)

    # ------------------------------
    # Prepare JSON Payload
    # ------------------------------
    payload = {
        "tool": "DnCNN",
        "tool_version": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        },
        "image": {
            "original_path": image_path,
            "file_name": os.path.basename(image_path),
            "original_shape": list(img.size[::-1]),
            "enhanced_shape": list(out_img.shape),
            "dtype": str(out_img.dtype),
        },
        "dncnn_parameters_used": {
            "sigma": sigma,
            "model_loaded": os.path.basename(model_path),
        },
    }

    # ------------------------------
    # Output Paths
    # ------------------------------
    base = os.path.splitext(os.path.basename(image_path))[0]
    unique_id = uuid.uuid4().hex[:8]
    stem = f"{base}_dncnn_{unique_id}"

    out_root_abs = os.path.abspath(out_root or ".")
    algo_dir = os.path.join(out_root_abs, "features", "dncnn_outputs")
    os.makedirs(algo_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(algo_dir, stem + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    # Save Image
    vis_path = os.path.join(algo_dir, stem + "_vis.jpg")
    Image.fromarray(out_img).save(vis_path)

    return os.path.abspath(json_path), os.path.abspath(vis_path)
