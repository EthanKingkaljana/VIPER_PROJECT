import os
import json
import numpy as np
import pytest
import torch
from PIL import Image

import server.algos.segmentation.deeplab_adapter as deeplab


# -------------------------------------------------
# Helper: create synthetic test image
# -------------------------------------------------
def create_test_image(tmp_path):

    img_path = tmp_path / "test_img.jpg"

    img = np.random.randint(
        0, 255, (128, 128, 3), dtype=np.uint8
    )

    Image.fromarray(img).save(img_path)

    return str(img_path)


# -------------------------------------------------
# Dummy segmentation model
# -------------------------------------------------
class DummyModel(torch.nn.Module):

    def forward(self, x):

        b, c, h, w = x.shape

        # create fake segmentation logits
        logits = torch.zeros((1, 21, h, w))

        # pretend class 15 ("person") detected
        logits[:, 15, :, :] = 10

        return {"out": logits}


# -------------------------------------------------
# Fixture: mock DeepLab model
# -------------------------------------------------
@pytest.fixture
def mock_model(monkeypatch):

    dummy = DummyModel()

    monkeypatch.setattr(
        deeplab,
        "get_model",
        lambda device, model_path=None: dummy
    )


# -------------------------------------------------
# Test: basic run
# -------------------------------------------------
def test_deeplab_basic(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    json_path, mask_path, vis_path = deeplab.run(
        img_path,
        tmp_path
    )

    assert os.path.exists(json_path)
    assert os.path.exists(mask_path)
    assert os.path.exists(vis_path)


# -------------------------------------------------
# Test: segmentation mask validity
# -------------------------------------------------
def test_mask_output(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    _, mask_path, _ = deeplab.run(
        img_path,
        tmp_path
    )

    mask = Image.open(mask_path)

    assert mask is not None
    assert mask.mode == "RGB"


# -------------------------------------------------
# Test: overlay visualization
# -------------------------------------------------
def test_overlay_output(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    _, _, vis_path = deeplab.run(
        img_path,
        tmp_path
    )

    overlay = Image.open(vis_path)

    assert overlay is not None
    assert overlay.mode == "RGB"


# -------------------------------------------------
# Test: JSON metadata structure
# -------------------------------------------------
def test_json_metadata(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    json_path, _, _ = deeplab.run(
        img_path,
        tmp_path
    )

    with open(json_path) as f:
        data = json.load(f)

    assert "detected_classes" in data
    assert "confidence_threshold" in data


# -------------------------------------------------
# Test: detected classes logic
# -------------------------------------------------
def test_detected_classes(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    json_path, _, _ = deeplab.run(
        img_path,
        tmp_path
    )

    with open(json_path) as f:
        data = json.load(f)

    assert isinstance(data["detected_classes"], list)
    assert "person" in data["detected_classes"]


# -------------------------------------------------
# Test: threshold parameter
# -------------------------------------------------
def test_threshold_parameter(tmp_path, mock_model):

    img_path = create_test_image(tmp_path)

    json_path, _, _ = deeplab.run(
        img_path,
        tmp_path,
        threshold=0.5
    )

    with open(json_path) as f:
        data = json.load(f)

    assert data["confidence_threshold"] == 0.5


# -------------------------------------------------
# Test: invalid image path
# -------------------------------------------------
def test_invalid_path(mock_model):

    with pytest.raises(Exception):
        deeplab.run("missing.jpg", ".")