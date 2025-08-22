import pytest
import numpy as np
import tempfile
import os
from pidng.dng import DNGTags, Tag
from pidng.defs import Orientation, PhotometricInterpretation, CFAPattern, CalibrationIlluminant


@pytest.fixture
def sample_image_data():
    """Fixture providing sample 16-bit image data for testing."""
    return np.array([
        [100, 200, 300, 400],
        [500, 600, 700, 800],
        [900, 1000, 1100, 1200]
    ], dtype=np.uint16)


@pytest.fixture
def sample_float_image_data():
    """Fixture providing sample float32 image data for testing."""
    return np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ], dtype=np.float32)


@pytest.fixture
def basic_dng_tags():
    """Fixture providing basic DNG tags for testing."""
    tags = DNGTags()
    tags.set(Tag.ImageWidth, 4)
    tags.set(Tag.ImageLength, 3)
    tags.set(Tag.BitsPerSample, 16)
    tags.set(Tag.Orientation, Orientation.Horizontal)
    tags.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    tags.set(Tag.SamplesPerPixel, 1)
    tags.set(Tag.CFARepeatPatternDim, [2, 2])
    tags.set(Tag.CFAPattern, CFAPattern.RGGB)
    tags.set(Tag.Make, "Test Camera")
    tags.set(Tag.Model, "Test Model")
    return tags


@pytest.fixture
def complete_dng_tags():
    """Fixture providing a complete set of DNG tags for realistic testing."""
    tags = DNGTags()

    # Basic image properties
    tags.set(Tag.ImageWidth, 1920)
    tags.set(Tag.ImageLength, 1080)
    tags.set(Tag.BitsPerSample, 12)
    tags.set(Tag.Orientation, Orientation.Horizontal)
    tags.set(Tag.SamplesPerPixel, 1)

    # Color filter array
    tags.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    tags.set(Tag.CFARepeatPatternDim, [2, 2])
    tags.set(Tag.CFAPattern, CFAPattern.RGGB)

    # Camera info
    tags.set(Tag.Make, "TestCorp")
    tags.set(Tag.Model, "TestCamera Pro")
    tags.set(Tag.Software, "PiDNG Test Suite")

    # Exposure and color
    tags.set(Tag.ExposureTime, [[1, 60]])  # 1/60 second
    tags.set(Tag.PhotographicSensitivity, [400])  # ISO 400
    tags.set(Tag.BlackLevel, [64, 64, 64, 64])
    tags.set(Tag.WhiteLevel, [4095])  # 12-bit max
    tags.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.D65)

    # Color matrix (identity-ish for testing)
    ccm1 = [[10000, 10000], [0, 10000], [0, 10000],
            [0, 10000], [10000, 10000], [0, 10000],
            [0, 10000], [0, 10000], [10000, 10000]]
    tags.set(Tag.ColorMatrix1, ccm1)
    tags.set(Tag.AsShotNeutral, [[1, 1], [1, 1], [1, 1]])

    return tags


@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def picamera2_format():
    """Fixture providing sample PiCamera2 format dictionary."""
    return {
        "format": "SRGGB10",
        "size": (1920, 1080)
    }


@pytest.fixture
def picamera2_metadata():
    """Fixture providing sample PiCamera2 metadata dictionary."""
    return {
        "SensorBlackLevels": [64, 64, 64, 64],
        "ColourGains": [1.2, 1.8],
        "ColourCorrectionMatrix": [
            1.5, -0.3, -0.2,
            -0.4, 1.3, 0.1,
            0.0, -0.5, 1.5
        ],
        "ExposureTime": 16667,  # ~1/60 second in microseconds
        "AnalogueGain": 2.0,
        "DigitalGain": 1.5,
        "SensorTimestamp": 1234567890123
    }


@pytest.fixture
def sample_packed_data():
    """Fixture providing sample packed data for testing unpacking functions."""
    return {
        "10bit": np.array([[0xFF, 0x00, 0x80, 0x40, 0x20]], dtype=np.uint8),
        "12bit": np.array([[0xFF, 0x0F, 0xF0]], dtype=np.uint8),
        "14bit": np.array([[0xFF, 0x3F, 0xFC, 0x0F, 0xF0, 0x03, 0xC0]], dtype=np.uint8)
    }


# Test data generators for parameterized tests
@pytest.fixture(params=[
    (1920, 1080, 10),
    (2560, 1440, 12),
    (4096, 3072, 14),
    (640, 480, 16)
])
def image_dimensions(request):
    """Parameterized fixture providing various image dimensions and bit depths."""
    width, height, bpp = request.param
    return {
        "width": width,
        "height": height,
        "bpp": bpp,
        "data": np.random.randint(0, (1 << bpp) - 1, (height, width), dtype=np.uint16)
    }


@pytest.fixture(params=[
    CFAPattern.RGGB,
    CFAPattern.BGGR,
    CFAPattern.GBRG,
    CFAPattern.GRBG
])
def cfa_patterns(request):
    """Parameterized fixture providing different CFA patterns."""
    return request.param


@pytest.fixture(params=[
    "SRGGB10", "SRGGB12", "SRGGB16",
    "SBGGR10", "SBGGR12", "SBGGR16",
    "SGBRG10", "SGBRG12", "SGBRG16",
    "SGRBG10", "SGRBG12", "SGRBG16"
])
def format_strings(request):
    """Parameterized fixture providing different camera format strings."""
    return request.param


# Mock data for testing without external dependencies
@pytest.fixture
def mock_ljpeg_compress():
    """Mock fixture for ljpeg compression to avoid external dependency."""
    def mock_pack16tolj(data, width, height, bpp, a, b, c, d, e):
        # Return a simple mock compressed data
        return b"MOCK_LJPEG_DATA" + data.tobytes()[:100]  # Truncated for testing

    return mock_pack16tolj


# Utility fixtures for error testing
@pytest.fixture
def invalid_image_data():
    """Fixture providing various invalid image data for error testing."""
    return {
        "wrong_dtype": np.array([[1, 2], [3, 4]], dtype=np.uint8),
        "wrong_shape": np.array([1, 2, 3, 4], dtype=np.uint16),
        "empty": np.array([], dtype=np.uint16),
        "non_array": [[1, 2], [3, 4]]
    }


@pytest.fixture
def incomplete_tags():
    """Fixture providing incomplete tag sets for error testing."""
    return {
        "no_width": lambda: DNGTags().set(Tag.ImageLength, 100).set(Tag.BitsPerSample, 16),
        "no_height": lambda: DNGTags().set(Tag.ImageWidth, 100).set(Tag.BitsPerSample, 16),
        "no_bpp": lambda: DNGTags().set(Tag.ImageWidth, 100).set(Tag.ImageLength, 100)
    }
