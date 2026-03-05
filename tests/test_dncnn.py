import os
import json
import numpy as np
import pytest
import torch
from PIL import Image

import server.algos.restoration.dncnn_adapter as dncnn


"""
Full validation tests for DnCNN adapter.

Tests verify:
- pipeline execution
- image input/output correctness
- metadata structure
- parameter propagation
- error handling

The real DnCNN model is mocked to avoid loading heavy weights.
"""


# -------------------------------------------------
# Helper: create synthetic grayscale image
# -------------------------------------------------
def create_gray_image(tmp_path):

    img_path = tmp_path / "test_img.jpg"

    img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    Image.fromarray(img).save(img_path)

    return str(img_path)


# -------------------------------------------------
# Dummy model (bypass real weights)
# -------------------------------------------------
class DummyModel(torch.nn.Module):

    def forward(self, x):
        return x


@pytest.fixture
def mock_model(monkeypatch):

    dummy = DummyModel()

    monkeypatch.setattr(dncnn, "DnCNN", lambda channels=1: dummy)


# -------------------------------------------------
# Test 1: Basic pipeline run
# -------------------------------------------------
def test_dncnn_basic(tmp_path, mock_model):

    img_path = create_gray_image(tmp_path)

    json_path, vis_path = dncnn.run(img_path, tmp_path, sigma=25)

    assert os.path.exists(json_path)
    assert os.path.exists(vis_path)

    with open(json_path) as f:
        data = json.load(f)

    assert data["tool"] == "DnCNN"
    assert data["dncnn_parameters_used"]["sigma"] == 25


# -------------------------------------------------
# Test 2: Output image validity
# -------------------------------------------------
def test_dncnn_output_image(tmp_path, mock_model):

    img_path = create_gray_image(tmp_path)

    _, vis_path = dncnn.run(img_path, tmp_path)

    img = Image.open(vis_path)

    assert img is not None
    assert img.mode in ["L", "RGB"]


# -------------------------------------------------
# Test 3: Output dimensions match input
# -------------------------------------------------
def test_output_dimensions(tmp_path, mock_model):

    img_path = create_gray_image(tmp_path)

    _, vis_path = dncnn.run(img_path, tmp_path)

    original = Image.open(img_path)
    output = Image.open(vis_path)

    assert original.size == output.size


# -------------------------------------------------
# Test 4: Sigma parameter propagation
# -------------------------------------------------
def test_sigma_parameter(tmp_path, mock_model):

    img_path = create_gray_image(tmp_path)

    json_path, _ = dncnn.run(img_path, tmp_path, sigma=50)

    with open(json_path) as f:
        data = json.load(f)

    assert data["dncnn_parameters_used"]["sigma"] == 50


# -------------------------------------------------
# Test 5: JSON metadata structure
# -------------------------------------------------
def test_json_metadata(tmp_path, mock_model):

    img_path = create_gray_image(tmp_path)

    json_path, _ = dncnn.run(img_path, tmp_path)

    with open(json_path) as f:
        data = json.load(f)

    assert "tool" in data
    assert "tool_version" in data
    assert "image" in data
    assert "dncnn_parameters_used" in data


# -------------------------------------------------
# Test 6: Invalid image path
# -------------------------------------------------
def test_invalid_path(mock_model):

    with pytest.raises(Exception):
        dncnn.run("missing.jpg", ".")