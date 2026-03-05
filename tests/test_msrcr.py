import os
import json
import cv2
import numpy as np
import pytest

from server.algos.enchancement.msrcr_adapter import run as msrcr_run


# =================================================
# 1. FIXTURES & HELPERS
# =================================================

@pytest.fixture
def tmpdir(tmp_path):
    return tmp_path


def create_color_image(tmp_path, channels=3):
    """Generate synthetic images for testing"""

    img_path = tmp_path / "test_img.png"

    if channels == 3:
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    elif channels == 4:  # RGBA
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)

    elif channels == 1:  # Grayscale
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    cv2.imwrite(str(img_path), img)

    return str(img_path)


# =================================================
# 2. CORE FUNCTIONALITY TESTS
# =================================================

def test_msrcr_basic(tmpdir):
    """MS01: Basic MSRCR execution"""

    img_path = create_color_image(tmpdir, channels=3)

    json_path, vis_path = msrcr_run(img_path, tmpdir)

    assert os.path.exists(json_path)
    assert os.path.exists(vis_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["tool"] == "MSRCR"
    assert "msrcr_parameters_used" in data
    assert "image" in data


# =================================================
# 3. INPUT TYPE TESTS
# =================================================

def test_rgba_input(tmpdir):
    """MS02: Adapter should accept RGBA input"""

    img_path = create_color_image(tmpdir, channels=4)

    json_path, vis_path = msrcr_run(img_path, tmpdir)

    assert os.path.exists(vis_path)

    img = cv2.imread(vis_path)

    assert img is not None
    assert img.shape[2] == 3   # should convert to BGR


def test_grayscale_input_error(tmpdir):
    """MS03: MSRCR should reject grayscale images"""

    img_path = create_color_image(tmpdir, channels=1)

    with pytest.raises(ValueError):
        msrcr_run(img_path, tmpdir)


# =================================================
# 4. PARAMETER VALIDATION
# =================================================

def test_msrcr_parameters(tmpdir):
    """MS04: Custom parameters should be applied"""

    img_path = create_color_image(tmpdir)

    json_path, _ = msrcr_run(
        img_path,
        tmpdir,
        sigma_list=(10, 50, 200),
        G=4,
        b=20,
        alpha=120,
        beta=40
    )

    with open(json_path) as f:
        data = json.load(f)

    params = data["msrcr_parameters_used"]

    assert tuple(params["sigma_list"]) == (10, 50, 200)
    assert params["G"] == 4
    assert params["b"] == 20
    assert params["alpha"] == 120
    assert params["beta"] == 40


# =================================================
# 5. OUTPUT VALIDATION
# =================================================

def test_output_image_valid(tmpdir):
    """MS05: Output image should be readable"""

    img_path = create_color_image(tmpdir)

    _, vis_path = msrcr_run(img_path, tmpdir)

    img = cv2.imread(vis_path)

    assert img is not None
    assert img.shape[0] > 0
    assert img.shape[1] > 0
    assert img.shape[2] == 3


# =================================================
# 6. ERROR HANDLING
# =================================================

def test_invalid_image_path():
    """MS06: Invalid path should raise ValueError"""

    with pytest.raises(ValueError):
        msrcr_run("non_existent_image.jpg", ".")