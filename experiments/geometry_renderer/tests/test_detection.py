from io import BytesIO
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from experiments.geometry_renderer.detection import (MAX_BYTES, PortraitError, decode_image,
    detect_landmarks, select_face)
from experiments.geometry_renderer.session import DemoSession
from .conftest import FIXTURE


def encoded(image, fmt='PNG'):
    stream = BytesIO(); Image.fromarray(image).save(stream, format=fmt); return stream.getvalue()


def test_real_offline_mediapipe_one_face(licensed_image, licensed_landmarks, monkeypatch):
    import socket
    def forbidden(*args, **kwargs):
        raise AssertionError('No networking allowed during detection')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    points = detect_landmarks(licensed_image)
    assert points.shape == (468, 2)
    np.testing.assert_allclose(points, licensed_landmarks)


def test_real_no_face():
    with pytest.raises(PortraitError, match='No usable face'):
        detect_landmarks(np.full((256, 256, 3), 255, np.uint8))


def test_real_multiple_faces(licensed_image, licensed_landmarks):
    # Enlarge the two faces relative to the canvas; the short-range detector
    # does not reliably detect two full-body portraits at half frame width.
    low = np.floor(licensed_landmarks.min(axis=0)).astype(int)
    high = np.ceil(licensed_landmarks.max(axis=0)).astype(int)
    margin = ((high-low)*.4).astype(int)
    low = np.maximum(low-margin, 0)
    high = np.minimum(high+margin, [licensed_image.shape[1], licensed_image.shape[0]])
    head = licensed_image[low[1]:high[1],low[0]:high[0]]
    pair = np.concatenate((head, head), axis=1)
    with pytest.raises(PortraitError, match='Multiple faces'):
        detect_landmarks(pair)


def test_clipped_and_small_faces():
    p = SimpleNamespace(x=.5, y=.5)
    with pytest.raises(PortraitError, match='clipped or too small'):
        select_face([SimpleNamespace(landmark=[p]*468)], 256, 256)

@pytest.mark.parametrize('data,suffix', [(b'', '.png'), (b'bad', '.png'),
    (b'x'*(MAX_BYTES+1), '.jpg'), (b'abc', '.gif')])
def test_bad_files(data, suffix):
    with pytest.raises(PortraitError):
        decode_image(data, suffix)


def test_mismatched_format_and_small_image():
    pixels = np.zeros((64,64,3), np.uint8)
    with pytest.raises(PortraitError, match='do not match'):
        decode_image(encoded(pixels), '.jpg')
    with pytest.raises(PortraitError, match='at least 64'):
        decode_image(encoded(pixels[:32]), '.png')


def test_pixel_limit_and_resize():
    # Uniform data compresses cheaply; bounds checked before decoding pixels.
    with pytest.raises(PortraitError, match='16 megapixels'):
        decode_image(encoded(np.zeros((4001,4001), np.uint8)), '.png')
    image = decode_image(encoded(np.zeros((1200,1000), np.uint8)), '.png')
    assert image.shape == (960,800,3)


def test_exif_orientation_and_jpeg():
    buffer = BytesIO()
    image = Image.new('RGB', (80,100))
    exif = Image.Exif(); exif[274] = 6
    image.save(buffer, format='JPEG', exif=exif)
    assert decode_image(buffer.getvalue(), '.jpeg').shape == (80,100,3)


def test_session_reuses_landmarks_and_triangulation(licensed_landmarks):
    count = 0
    def detector(image):
        nonlocal count
        count += 1
        return licensed_landmarks
    session = DemoSession()
    data = FIXTURE.read_bytes()
    first = session.load(data, '.png', detector=detector)
    assert session.load(data, '.png', detector=detector) is first
    assert count == 1
    # Invalid new content must not leave the previous portrait in the session.
    with pytest.raises(PortraitError):
        session.load(b'bad', '.png', detector=detector)
    assert session.prepared is None
    session.load(data, '.png', detector=detector)
    assert count == 2
    session.clear()
    assert session.prepared is None and session._key is None
