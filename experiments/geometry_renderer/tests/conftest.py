from pathlib import Path
import sys
import numpy as np
import pytest

# Also support the issue's exact command when pytest sets a nested rootdir.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.geometry_renderer.core import REGIONS, prepare_face
from experiments.geometry_renderer.detection import decode_image, detect_landmarks

FIXTURE = Path(__file__).parent/'fixtures/astronaut.png'

@pytest.fixture(scope='session')
def licensed_image():
    return decode_image(FIXTURE.read_bytes(), '.png')

@pytest.fixture(scope='session')
def licensed_landmarks(licensed_image):
    return detect_landmarks(licensed_image)

@pytest.fixture(scope='session')
def licensed_prepared(licensed_image, licensed_landmarks):
    return prepare_face(licensed_image, licensed_landmarks)

@pytest.fixture
def synthetic():
    rng = np.random.default_rng(4)
    points = rng.uniform([40, 25], [120, 135], (468, 2))
    for region, centers in {
        'nose': [(80, 70), (80, 74), (80, 67)],
        'cheeks': [(58, 82), (60, 85), (62, 88), (102, 82), (100, 85), (98, 88)],
        'jowls': [(52, 108), (57, 119), (64, 126), (108, 108), (103, 119), (96, 126)],
    }.items():
        for index, center in zip(REGIONS[region], centers):
            points[index] = center
    y, x = np.indices((160, 160))
    image = np.stack((x, y, (x+y)//2), axis=-1).astype(np.uint8)
    return prepare_face(image, points)
