import os
import json
import cv2
import numpy as np
import pytest
import torch

import server.algos.enchancement.zero_adapter as Zero


"""
Full validation tests for Zero-DCE adapter.

Validates:
- pipeline execution
- model inference (mocked)
- output image correctness
- parameter propagation
- metadata structure
- error handling
"""


# -------------------------------------------------
# Helper: create synthetic RGB image
# -------------------------------------------------
def create_color_image(tmp_path):

    img_path = tmp_path / "test_img.jpg"

    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    return str(img_path)


# -------------------------------------------------
# Dummy model (avoid loading real weights)
# -------------------------------------------------
class DummyModel(torch.nn.Module):

    def forward(self, x, num_iters=8):
        return torch.clamp(x + 0.1, 0, 1)


@pytest.fixture
def mock_model(monkeypatch):

    dummy = DummyModel()

    monkeypatch.setattr(Zero, "load_model", lambda path: dummy)


# -------------------------------------------------
# Test 1: Basic pipeline execution
# -------------------------------------------------
def test_zero_basic(tmp_path, mock_model):

    img_path = create_color_image(tmp_path)

    json_path, vis_path = Zero.run(img_path, tmp_path, iterations=4)

    assert os.path.exists(json_path)
    assert os.path.exists(vis_path)

    with open(json_path) as f:
        data = json.load(f)

    assert data["tool"] == "Zero-DCE"
    assert data["parameters"]["iterations"] == 4


# -------------------------------------------------
# Test 2: Output image validity
# -------------------------------------------------
def test_zero_output_image(tmp_path, mock_model):

    img_path = create_color_image(tmp_path)

    _, vis_path = Zero.run(img_path, tmp_path)

    img = cv2.imread(vis_path)

    assert img is not None
    assert img.shape[2] == 3


# -------------------------------------------------
# Test 3: Output dimensions match input
# -------------------------------------------------
def test_output_dimensions(tmp_path, mock_model):

    img_path = create_color_image(tmp_path)

    _, vis_path = Zero.run(img_path, tmp_path)

    original = cv2.imread(img_path)
    output = cv2.imread(vis_path)

    assert original.shape == output.shape


# -------------------------------------------------
# Test 4: Parameter propagation
# -------------------------------------------------
def test_iteration_parameter(tmp_path, mock_model):

    img_path = create_color_image(tmp_path)

    json_path, _ = Zero.run(img_path, tmp_path, iterations=10)

    with open(json_path) as f:
        data = json.load(f)

    assert data["parameters"]["iterations"] == 10


# -------------------------------------------------
# Test 5: JSON metadata schema
# -------------------------------------------------
def test_json_metadata(tmp_path, mock_model):

    img_path = create_color_image(tmp_path)

    json_path, _ = Zero.run(img_path, tmp_path)

    with open(json_path) as f:
        data = json.load(f)

    assert "tool" in data
    assert "tool_version" in data
    assert "image" in data
    assert "parameters" in data
    assert "device" in data


# -------------------------------------------------
# Test 6: Invalid image path
# -------------------------------------------------
def test_invalid_path(mock_model):

    with pytest.raises(ValueError):
        Zero.run("missing.jpg", ".")