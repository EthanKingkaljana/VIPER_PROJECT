import os
import json
import uuid
import torch
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

# ✅ Correct relative import (VERY IMPORTANT)
from .network_swinir import SwinIR


# ============================================================
# SWINIR ADAPTER
# ============================================================

def run(image_path: str, out_root: str = ".", model_name: str = None, **params):
    """
    image_path  : input image path
    out_root    : output root directory
    model_name  : filename inside weights folder
    params:
        scale   : 2 / 3 / 4 (must match model)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale = int(params.get("scale", 4))

    # Safe tiling for VRAM stability
    tile_size = 128
    tile_pad = 16

    # --------------------------------------------------------
    # Resolve weights path
    # --------------------------------------------------------
    base_dir = os.path.dirname(__file__)
    server_root = os.path.abspath(os.path.join(base_dir, "..", ".."))
    weights_dir = os.path.join(server_root, "weights")

    if not model_name:
        model_name = "003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"

    model_path = os.path.join(weights_dir, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # --------------------------------------------------------
    # Initialize OFFICIAL SwinIR
    # --------------------------------------------------------
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="3conv",
).to(device)

    # Load weights correctly
    pretrained = torch.load(model_path, map_location=device)
    if "params" in pretrained:
        pretrained = pretrained["params"]

    model.load_state_dict(pretrained, strict=True)
    model.eval()

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------
    img = Image.open(image_path).convert("RGB")
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    b, c, h, w = img_tensor.size()

    h_out = h * scale
    w_out = w * scale
    output = torch.zeros((b, c, h_out, w_out), device=device)

    # --------------------------------------------------------
    # Tiled Inference
    # --------------------------------------------------------
    with torch.no_grad():
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):

                y1 = max(0, y - tile_pad)
                y2 = min(h, y + tile_size + tile_pad)
                x1 = max(0, x - tile_pad)
                x2 = min(w, x + tile_size + tile_pad)

                tile = img_tensor[:, :, y1:y2, x1:x2]

                out_tile = model(tile)

                # Destination (scaled)
                dest_y1 = y * scale
                dest_y2 = (y + min(tile_size, h - y)) * scale
                dest_x1 = x * scale
                dest_x2 = (x + min(tile_size, w - x)) * scale

                # Remove padding (scaled)
                src_y1 = (y - y1) * scale
                src_y2 = (y - y1 + min(tile_size, h - y)) * scale
                src_x1 = (x - x1) * scale
                src_x2 = (x - x1 + min(tile_size, w - x)) * scale

                output[:, :, dest_y1:dest_y2, dest_x1:dest_x2] = \
                    out_tile[:, :, src_y1:src_y2, src_x1:src_x2]

    # --------------------------------------------------------
    # Convert to image
    # --------------------------------------------------------
    out_img = output.squeeze().cpu().clamp(0, 1).numpy()
    out_img = np.transpose(out_img, (1, 2, 0))
    out_img = (out_img * 255.0).round().astype(np.uint8)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    payload = {
        "tool": "SwinIR",
        "image": {
            "original_path": image_path,
            "original_shape": [h, w],
            "enhanced_shape": [h_out, w_out, c],
        },
        "parameters_used": {
            "scale": scale,
            "tiling": True,
            "tile_size": tile_size,
        }
    }

    base = os.path.splitext(os.path.basename(image_path))[0]
    uid = uuid.uuid4().hex[:8]
    stem = f"{base}_swinir_{uid}"

    out_root_abs = os.path.abspath(out_root or ".")
    algo_dir = os.path.join(out_root_abs, "features", "swinir_outputs")
    os.makedirs(algo_dir, exist_ok=True)

    json_path = os.path.join(algo_dir, stem + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    vis_path = os.path.join(algo_dir, stem + "_vis.jpg")
    Image.fromarray(out_img).save(vis_path)

    return os.path.abspath(json_path), os.path.abspath(vis_path)


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    test_img = "lele.jpg"

    json_file, img_file = run(
        image_path=test_img,
        out_root=".",
        model_name="003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth",
        scale=4
    )
    
    print("JSON:", json_file)
    print("IMAGE:", img_file)