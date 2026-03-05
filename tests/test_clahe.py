import os
import json
import cv2
import numpy as np
import pytest

from server.algos.enchancement.clahe_adapter import run as clahe_run


# =================================================
# 1. FIXTURES & HELPERS
# =================================================

@pytest.fixture
def tmpdir(tmp_path):
    return tmp_path


def create_test_image(tmp_path, channels=3):
    """Generate synthetic test images."""

    img_path = tmp_path / "test_img.png"

    if channels == 1:
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

    elif channels == 3:
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    elif channels == 4:
        img = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)

    cv2.imwrite(str(img_path), img)

    return str(img_path)


# =================================================
# 2. CORE FUNCTIONALITY TESTS
# =================================================

def test_clahe_basic(tmpdir):
    """CL01: Basic CLAHE execution"""

    img_path = create_test_image(tmpdir, channels=3)

    json_path, vis_path = clahe_run(img_path, tmpdir)

    assert os.path.exists(json_path)
    assert os.path.exists(vis_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    assert data["tool"] == "CLAHE"
    assert "clahe_parameters_used" in data
    assert "image" in data


# =================================================
# 3. INPUT TYPE TESTS
# =================================================

def test_grayscale_input(tmpdir):
    """CL02: Accept grayscale images"""

    img_path = create_test_image(tmpdir, channels=1)

    json_path, vis_path = clahe_run(img_path, tmpdir)

    img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)

    assert os.path.exists(json_path)
    assert img is not None
    assert len(img.shape) == 2


def test_rgba_input(tmpdir):
    """CL03: Accept RGBA images"""

    img_path = create_test_image(tmpdir, channels=4)

    json_path, vis_path = clahe_run(img_path, tmpdir)

    img = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)

    assert os.path.exists(vis_path)
    assert img is not None


# =================================================
# 4. PARAMETER TESTS
# =================================================

def test_clahe_parameters(tmpdir):
    """CL04: CLAHE parameter override"""

    img_path = create_test_image(tmpdir)

    json_path, _ = clahe_run(
        img_path,
        tmpdir,
        clipLimit=2.5,
        tileGridSize=(4, 4)
    )

    with open(json_path) as f:
        data = json.load(f)

    params = data["clahe_parameters_used"]

    assert params["clipLimit"] == 2.5
    assert tuple(params["tileGridSize"]) == (4, 4)


def test_tilegrid_string(tmpdir):
    """CL05: tileGridSize supports string format"""

    img_path = create_test_image(tmpdir)

    json_path, _ = clahe_run(
        img_path,
        tmpdir,
        tileGridSize="8,8"
    )

    with open(json_path) as f:
        data = json.load(f)

    params = data["clahe_parameters_used"]

    assert tuple(params["tileGridSize"]) == (8, 8)


# =================================================
# 5. OUTPUT VALIDATION
# =================================================

def test_output_image_exists(tmpdir):
    """CL06: Output image is saved"""

    img_path = create_test_image(tmpdir)

    _, vis_path = clahe_run(img_path, tmpdir)

    img = cv2.imread(vis_path)

    assert img is not None
    assert img.shape[0] > 0
    assert img.shape[1] > 0


# =================================================
# 6. ERROR HANDLING
# =================================================

def test_invalid_image_path():
    """CL07: Invalid image path should raise error"""

    with pytest.raises(ValueError):
        clahe_run("non_existent_file.jpg", ".")