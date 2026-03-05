import os
import json
import cv2
import numpy as np
import pytest

import server.algos.restoration.real_adapter as real


"""
Full validation tests for Real-ESRGAN adapter.

Validates:
- pipeline execution
- model inference (mocked)
- resolution scaling
- parameter propagation
- metadata schema
- error handling
"""


# -------------------------------------------------
# Helper: create synthetic RGB image
# -------------------------------------------------
def create_test_image(tmp_path):

    img_path = tmp_path / "test_img.jpg"

    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    return str(img_path)


# -------------------------------------------------
# Dummy RealESRGAN model
# -------------------------------------------------
class DummyUpsampler:

    def enhance(self, img, outscale=4):

        h, w, _ = img.shape

        out = cv2.resize(
            img,
            (int(w * outscale), int(h * outscale))
        )

        return out, None


@pytest.fixture
def mock_realesrgan(monkeypatch):

    monkeypatch.setattr(
        real,
        "RealESRGANer",
        lambda *args, **kwargs: DummyUpsampler()
    )


# -------------------------------------------------
# Helper: create fake weights
# -------------------------------------------------
def create_fake_weights(tmp_path):

    weight_file = tmp_path / "RealESRGAN_x4plus.pth"
    weight_file.write_text("dummy")

    return str(weight_file)


# -------------------------------------------------
# Test 1: Basic pipeline execution
# -------------------------------------------------
def test_real_basic(tmp_path, mock_realesrgan):

    img_path = create_test_image(tmp_path)
    weight_file = create_fake_weights(tmp_path)

    json_path, out_path = real.run(
        img_path,
        tmp_path,
        model_path=weight_file,
        scale=2
    )

    assert os.path.exists(json_path)
    assert os.path.exists(out_path)

    with open(json_path) as f:
        data = json.load(f)

    assert data["tool"] == "Real-ESRGAN"
    assert data["parameters_used"]["scale"] == 2


# -------------------------------------------------
# Test 2: Resolution scaling
# -------------------------------------------------
def test_output_resolution(tmp_path, mock_realesrgan):

    img_path = create_test_image(tmp_path)
    weight_file = create_fake_weights(tmp_path)

    _, out_path = real.run(
        img_path,
        tmp_path,
        model_path=weight_file,
        scale=2
    )

    img = cv2.imread(out_path)

    assert img.shape[0] > 64
    assert img.shape[1] > 64


# -------------------------------------------------
# Test 3: RGB output validation
# -------------------------------------------------
def test_output_channels(tmp_path, mock_realesrgan):

    img_path = create_test_image(tmp_path)
    weight_file = create_fake_weights(tmp_path)

    _, out_path = real.run(
        img_path,
        tmp_path,
        model_path=weight_file
    )

    img = cv2.imread(out_path)

    assert img is not None
    assert img.shape[2] == 3


# -------------------------------------------------
# Test 4: Parameter propagation
# -------------------------------------------------
def test_scale_parameter(tmp_path, mock_realesrgan):

    img_path = create_test_image(tmp_path)
    weight_file = create_fake_weights(tmp_path)

    json_path, _ = real.run(
        img_path,
        tmp_path,
        model_path=weight_file,
        scale=3
    )

    with open(json_path) as f:
        data = json.load(f)

    assert data["parameters_used"]["scale"] == 3


# -------------------------------------------------
# Test 5: JSON metadata schema
# -------------------------------------------------
def test_json_metadata(tmp_path, mock_realesrgan):

    img_path = create_test_image(tmp_path)
    weight_file = create_fake_weights(tmp_path)

    json_path, _ = real.run(
        img_path,
        tmp_path,
        model_path=weight_file
    )

    with open(json_path) as f:
        data = json.load(f)

    assert "tool" in data
    assert "parameters_used" in data
    assert "device" in data
    assert "input_resolution" in data
    assert "output_resolution" in data


# -------------------------------------------------
# Test 6: Invalid image path
# -------------------------------------------------
def test_invalid_path(tmp_path, mock_realesrgan):

    weight_file = create_fake_weights(tmp_path)

    with pytest.raises(ValueError):
        real.run("missing.jpg", tmp_path, model_path=weight_file)


# -------------------------------------------------
# Test 7: Missing weights
# -------------------------------------------------
def test_missing_weights(tmp_path, mock_realesrgan):

    img_path = create_test_image(tmp_path)

    with pytest.raises(Exception):
        real.run(
            img_path,
            tmp_path,
            model_path="missing_weights.pth"
        )