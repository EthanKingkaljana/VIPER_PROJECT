import os
import json
import numpy as np
import pytest
import torch
from PIL import Image

import server.algos.segmentation.maskrcnn_adapter as maskrcnn


# -------------------------------------------------
# Helper: create synthetic RGB image
# -------------------------------------------------
def create_test_image(tmp_path):

    img_path = tmp_path / "test_img.jpg"

    img = np.random.randint(
        0, 255, (128, 128, 3), dtype=np.uint8
    )

    Image.fromarray(img).save(img_path)

    return str(img_path)


# -------------------------------------------------
# Dummy MaskRCNN model
# -------------------------------------------------
class DummyModel(torch.nn.Module):

    def forward(self, images):

        device = images[0].device
        H, W = images[0].shape[1:]

        boxes = torch.tensor([[10, 10, 60, 60]], device=device).float()

        masks = torch.zeros((1, 1, H, W), device=device)
        masks[:, :, 20:80, 20:80] = 1

        labels = torch.tensor([1], device=device)  # person
        scores = torch.tensor([0.9], device=device)

        return [{
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks
        }]


# -------------------------------------------------
# Fixture: mock MaskRCNN model
# -------------------------------------------------
@pytest.fixture
def mock_model(monkeypatch):

    dummy = DummyModel()

    monkeypatch.setattr(
        maskrcnn,
        "load_model",
        lambda device: dummy
    )


# -------------------------------------------------
# Test: basic pipeline run
# -------------------------------------------------
def test_maskrcnn_basic(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    json_path, mask_path, overlay_path, detection_path = maskrcnn.run(
        img_path,
        tmp_path
    )

    assert os.path.exists(json_path)
    assert os.path.exists(mask_path)
    assert os.path.exists(overlay_path)
    assert os.path.exists(detection_path)


# -------------------------------------------------
# Test: mask output validity
# -------------------------------------------------
def test_mask_output(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    _, mask_path, _, _ = maskrcnn.run(
        img_path,
        tmp_path
    )

    mask = Image.open(mask_path)

    assert mask is not None
    assert mask.mode in ["L", "RGB"]


# -------------------------------------------------
# Test: overlay output
# -------------------------------------------------
def test_overlay_output(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    _, _, overlay_path, _ = maskrcnn.run(
        img_path,
        tmp_path
    )

    overlay = Image.open(overlay_path)

    assert overlay is not None
    assert overlay.mode == "RGB"


# -------------------------------------------------
# Test: detection visualization
# -------------------------------------------------
def test_detection_visualization(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    _, _, _, detection_path = maskrcnn.run(
        img_path,
        tmp_path
    )

    detection = Image.open(detection_path)

    assert detection is not None
    assert detection.mode == "RGB"


# -------------------------------------------------
# Test: JSON metadata structure
# -------------------------------------------------
def test_json_metadata(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    json_path, _, _, _ = maskrcnn.run(
        img_path,
        tmp_path
    )

    with open(json_path) as f:
        data = json.load(f)

    assert "tool" in data
    assert "tool_version" in data
    assert "detections" in data
    assert "image" in data
    assert "maskrcnn_parameters_used" in data


# -------------------------------------------------
# Test: detection contents
# -------------------------------------------------
def test_detection_contents(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    json_path, _, _, _ = maskrcnn.run(
        img_path,
        tmp_path
    )

    with open(json_path) as f:
        data = json.load(f)

    detections = data["detections"]

    assert isinstance(detections, list)
    assert len(detections) >= 1

    det = detections[0]

    assert "class_id" in det
    assert "class_name" in det
    assert "confidence" in det


# -------------------------------------------------
# Test: score threshold parameter
# -------------------------------------------------
def test_score_threshold(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    json_path, _, _, _ = maskrcnn.run(
        img_path,
        tmp_path,
        score_thr=0.3
    )

    with open(json_path) as f:
        data = json.load(f)

    assert data["maskrcnn_parameters_used"]["score_thr"] == 0.3


# -------------------------------------------------
# Test: invalid image path
# -------------------------------------------------
def test_invalid_path(mock_model):

    with pytest.raises(Exception):
        maskrcnn.run("missing.jpg", ".")