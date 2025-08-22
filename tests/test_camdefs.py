import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from pidng.camdefs import BaseCameraModel, Picamera2Camera, RaspberryPiCameraModels
from pidng.dng import DNGTags, Tag
from pidng.defs import CFAPattern, PhotometricInterpretation, CalibrationIlluminant, Orientation


class TestRaspberryPiCameraModels:
    """Test RaspberryPi camera model constants."""

    def test_camera_model_names(self):
        """Test that camera model names are defined correctly."""
        assert RaspberryPiCameraModels.Raspberry_Pi_Camera_V1 == "Raspberry Pi Camera V1"
        assert RaspberryPiCameraModels.Raspberry_Pi_Camera_V2 == "Raspberry Pi Camera V2"
        assert RaspberryPiCameraModels.Raspberry_Pi_High_Quality_Camera == "Raspberry Pi High Quality Camera"


class TestBaseCameraModel:
    """Test BaseCameraModel base class."""

    def test_base_camera_model_init(self):
        """Test BaseCameraModel initialization."""
        camera = BaseCameraModel()

        assert hasattr(camera, 'tags')
        assert isinstance(camera.tags, DNGTags)
        assert camera.model == "BaseCameraModel"

    def test_base_camera_model_repr(self):
        """Test BaseCameraModel __repr__ method."""
        camera = BaseCameraModel()
        result = repr(camera)
        assert result == "BaseCameraModel(model='BaseCameraModel')"

    def test_base_camera_model_str(self):
        """Test BaseCameraModel __str__ method."""
        camera = BaseCameraModel()
        result = str(camera)
        assert result == "BaseCameraModel"


class TestPicamera2Camera:
    """Test Picamera2Camera class."""

    def create_test_format(self, format_str="SRGGB10", size=(1920, 1080)):
        """Helper to create test format dictionary."""
        return {
            "format": format_str,
            "size": size
        }

    def create_test_metadata(self):
        """Helper to create test metadata dictionary."""
        return {
            "SensorBlackLevels": [64, 64, 64, 64],
            "ColourGains": [1.5, 2.0],
            "ColourCorrectionMatrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "ExposureTime": 33333,  # microseconds
            "AnalogueGain": 2.0,
            "DigitalGain": 1.0,
            "SensorTimestamp": 123456789
        }

    def test_picamera2_camera_init(self):
        """Test Picamera2Camera initialization."""
        fmt = self.create_test_format()
        metadata = self.create_test_metadata()

        camera = Picamera2Camera(fmt, metadata)

        assert camera.model == "PiDNG / PiCamera2"
        assert camera.fmt == fmt
        assert camera.metadata == metadata
        assert isinstance(camera.tags, DNGTags)

    def test_picamera2_camera_custom_model(self):
        """Test Picamera2Camera with custom model name."""
        fmt = self.create_test_format()
        metadata = self.create_test_metadata()
        custom_model = "Custom Camera Model"

        camera = Picamera2Camera(fmt, metadata, model=custom_model)

        assert camera.model == custom_model

    def test_bpp_extraction_from_format(self):
        """Test bit depth extraction from format string."""
        test_cases = [
            ("SRGGB10", 10),
            ("SRGGB12", 12),
            ("SBGGR8", 8),
            ("SGRBG16", 16)
        ]

        metadata = self.create_test_metadata()

        for format_str, expected_bpp in test_cases:
            fmt = self.create_test_format(format_str)
            camera = Picamera2Camera(fmt, metadata)

            assert camera.fmt["bpp"] == expected_bpp

    def test_cfa_pattern_detection(self):
        """Test CFA pattern detection from format string."""
        test_cases = [
            ("SRGGB10", CFAPattern.RGGB),
            ("SBGGR10", CFAPattern.BGGR),
            ("SGBRG10", CFAPattern.GBRG),
            ("SGRBG10", CFAPattern.GRBG)
        ]

        metadata = self.create_test_metadata()

        for format_str, expected_pattern in test_cases:
            fmt = self.create_test_format(format_str)
            camera = Picamera2Camera(fmt, metadata)

            assert camera.cfaPattern == expected_pattern

    def test_cfa_pattern_none_for_mono(self):
        """Test that CFA pattern is None for mono sensors."""
        fmt = self.create_test_format("Y10")  # Mono format
        metadata = self.create_test_metadata()

        camera = Picamera2Camera(fmt, metadata)

        assert camera.cfaPattern is None

    def test_basic_tags_set(self):
        """Test that basic DNG tags are set correctly."""
        fmt = self.create_test_format("SRGGB10", (1920, 1080))
        metadata = self.create_test_metadata()

        camera = Picamera2Camera(fmt, metadata)

        # Check basic dimension tags
        assert camera.tags.get(Tag.ImageWidth).rawValue == [1920]
        assert camera.tags.get(Tag.ImageLength).rawValue == [1080]
        assert camera.tags.get(Tag.BitsPerSample).rawValue == [10]
        assert camera.tags.get(Tag.SamplesPerPixel).rawValue == [1]

        # Check camera info tags
        assert camera.tags.get(Tag.Make).rawValue == "RaspberryPi"
        assert camera.tags.get(Tag.Model).rawValue == "PiDNG / PiCamera2"

    def test_exposure_calculation(self):
        """Test exposure time calculation."""
        fmt = self.create_test_format()
        metadata = self.create_test_metadata()
        metadata["ExposureTime"] = 33333  # 33.333ms in microseconds

        camera = Picamera2Camera(fmt, metadata)

        exposure_tag = camera.tags.get(Tag.ExposureTime)
        assert exposure_tag is not None
        # Should be [1, 30] for 1/30 second (approximately)
        assert exposure_tag.rawValue[0][0] == 1
        assert exposure_tag.rawValue[0][1] == 30  # 1/0.033333 ≈ 30

    def test_iso_calculation(self):
        """Test ISO calculation from gains."""
        fmt = self.create_test_format()
        metadata = self.create_test_metadata()
        metadata["AnalogueGain"] = 2.0
        metadata["DigitalGain"] = 1.5

        camera = Picamera2Camera(fmt, metadata)

        iso_tag = camera.tags.get(Tag.PhotographicSensitivity)
        assert iso_tag is not None
        # ISO = (2.0 * 1.5) * 100 = 300
        assert iso_tag.rawValue == [300]

    def test_black_level_calculation(self):
        """Test black level calculation with bit depth scaling."""
        fmt = self.create_test_format("SRGGB10")
        metadata = self.create_test_metadata()
        metadata["SensorBlackLevels"] = [1024, 1024, 1024, 1024]  # 16-bit values

        camera = Picamera2Camera(fmt, metadata)

        black_level_tag = camera.tags.get(Tag.BlackLevel)
        assert black_level_tag is not None
        # Should be scaled from 16-bit to 10-bit: 1024 >> (16-10) = 1024 >> 6 = 16
        assert all(level == 16 for level in black_level_tag.rawValue)

    def test_white_level_calculation(self):
        """Test white level calculation."""
        fmt = self.create_test_format("SRGGB12")
        metadata = self.create_test_metadata()

        camera = Picamera2Camera(fmt, metadata)

        white_level_tag = camera.tags.get(Tag.WhiteLevel)
        assert white_level_tag is not None
        # For 12-bit: (1 << 12) - 1 = 4095
        assert white_level_tag.rawValue == [4095]

    def test_color_sensor_tags(self):
        """Test tags specific to color sensors."""
        fmt = self.create_test_format("SRGGB10")
        metadata = self.create_test_metadata()

        camera = Picamera2Camera(fmt, metadata)

        # Check color-specific tags
        assert camera.tags.get(Tag.PhotometricInterpretation).rawValue == [PhotometricInterpretation.Color_Filter_Array]
        assert camera.tags.get(Tag.CFARepeatPatternDim).rawValue == [2, 2]
        assert camera.tags.get(Tag.CFAPattern).rawValue == CFAPattern.RGGB
        assert camera.tags.get(Tag.BlackLevelRepeatDim).rawValue == [2, 2]

        # Check that color matrix is set
        assert camera.tags.get(Tag.ColorMatrix1) is not None
        assert camera.tags.get(Tag.CalibrationIlluminant1).rawValue == [CalibrationIlluminant.D65]

    def test_mono_sensor_tags(self):
        """Test tags specific to mono sensors."""
        fmt = self.create_test_format("Y10")  # Mono format
        metadata = self.create_test_metadata()

        camera = Picamera2Camera(fmt, metadata)

        # Check mono-specific tags
        assert camera.tags.get(Tag.PhotometricInterpretation).rawValue == [PhotometricInterpretation.Linear_Raw]
        assert camera.tags.get(Tag.BlackLevelRepeatDim).rawValue == [1, 1]

        # Color-specific tags should not be set for mono
        assert camera.tags.get(Tag.CFAPattern) is None
        assert camera.tags.get(Tag.ColorMatrix1) is None

    def test_as_shot_neutral_calculation(self):
        """Test as-shot neutral calculation from color gains."""
        fmt = self.create_test_format("SRGGB10")
        metadata = self.create_test_metadata()
        metadata["ColourGains"] = [1.2, 1.8]

        camera = Picamera2Camera(fmt, metadata)

        as_shot_tag = camera.tags.get(Tag.AsShotNeutral)
        assert as_shot_tag is not None

        # Should be inversely proportional to gains
        # Format: [[10000, gain_r_scaled], [10000, 10000], [10000, gain_b_scaled]]
        neutral = as_shot_tag.rawValue
        assert len(neutral) == 3
        assert neutral[0] == [10000, 12000]  # Red: 1.2 * 10000
        assert neutral[1] == [10000, 10000]  # Green: 1.0 * 10000
        assert neutral[2] == [10000, 18000]  # Blue: 1.8 * 10000

    def test_sensor_timestamp_handling(self):
        """Test sensor timestamp handling."""
        fmt = self.create_test_format()
        metadata = self.create_test_metadata()
        metadata["SensorTimestamp"] = 987654321

        camera = Picamera2Camera(fmt, metadata)

        timestamp_tag = camera.tags.get(Tag.RawDataUniqueID)
        assert timestamp_tag is not None
        assert timestamp_tag.rawValue == b"987654321"

    def test_missing_metadata_handling(self):
        """Test handling of missing metadata values."""
        fmt = self.create_test_format()
        metadata = {}  # Empty metadata

        # Should handle missing values gracefully with defaults
        camera = Picamera2Camera(fmt, metadata)

        # Should still create basic tags even with missing metadata
        assert camera.tags.get(Tag.ImageWidth) is not None
        assert camera.tags.get(Tag.ImageLength) is not None
        assert camera.tags.get(Tag.Make) is not None


class TestCameraIntegration:
    """Integration tests for camera classes."""

    def test_camera_tags_can_be_used_for_dng_creation(self):
        """Test that camera-generated tags are suitable for DNG creation."""
        fmt = {"format": "SRGGB10", "size": (1920, 1080)}
        metadata = {
            "SensorBlackLevels": [64, 64, 64, 64],
            "ColourGains": [1.0, 1.0],
            "ColourCorrectionMatrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "ExposureTime": 33333,
            "AnalogueGain": 1.0,
            "DigitalGain": 1.0,
            "SensorTimestamp": 123456789
        }

        camera = Picamera2Camera(fmt, metadata)

        # Check that all required tags for DNG creation are present
        required_tags = [
            Tag.ImageWidth, Tag.ImageLength, Tag.BitsPerSample,
            Tag.PhotometricInterpretation, Tag.Make, Tag.Model
        ]

        for tag in required_tags:
            assert camera.tags.get(tag) is not None, f"Required tag {tag} is missing"

    def test_different_camera_models_have_different_tags(self):
        """Test that different camera configurations produce different tag sets."""
        fmt1 = {"format": "SRGGB10", "size": (1920, 1080)}
        fmt2 = {"format": "SBGGR12", "size": (2560, 1440)}
        metadata = {
            "SensorBlackLevels": [64, 64, 64, 64],
            "ColourGains": [1.0, 1.0],
            "ColourCorrectionMatrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "ExposureTime": 33333,
            "AnalogueGain": 1.0,
            "DigitalGain": 1.0,
            "SensorTimestamp": 123456789
        }

        camera1 = Picamera2Camera(fmt1, metadata, "Camera1")
        camera2 = Picamera2Camera(fmt2, metadata, "Camera2")

        # Different resolutions
        assert camera1.tags.get(Tag.ImageWidth).rawValue != camera2.tags.get(Tag.ImageWidth).rawValue
        assert camera1.tags.get(Tag.ImageLength).rawValue != camera2.tags.get(Tag.ImageLength).rawValue

        # Different bit depths
        assert camera1.tags.get(Tag.BitsPerSample).rawValue != camera2.tags.get(Tag.BitsPerSample).rawValue

        # Different CFA patterns
        assert camera1.tags.get(Tag.CFAPattern).rawValue != camera2.tags.get(Tag.CFAPattern).rawValue

        # Different model names
        assert camera1.tags.get(Tag.Model).rawValue != camera2.tags.get(Tag.Model).rawValue
