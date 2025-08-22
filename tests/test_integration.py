import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from pidng.core import RAW2DNG
from pidng.dng import DNGTags, Tag
from pidng.defs import *
from pidng.camdefs import Picamera2Camera


class TestEndToEndWorkflow:
    """Integration tests for complete DNG creation workflow."""

    @pytest.mark.integration
    def test_complete_raw_to_dng_workflow(self, temp_directory):
        """Test complete workflow from raw data to DNG file."""
        # Create test raw data
        width, height = 64, 48
        raw_data = np.random.randint(0, 4095, (height, width), dtype=np.uint16)

        # Create DNG tags
        tags = DNGTags()
        tags.set(Tag.ImageWidth, width)
        tags.set(Tag.ImageLength, height)
        tags.set(Tag.BitsPerSample, 12)
        tags.set(Tag.Orientation, Orientation.Horizontal)
        tags.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
        tags.set(Tag.SamplesPerPixel, 1)
        tags.set(Tag.CFARepeatPatternDim, [2, 2])
        tags.set(Tag.CFAPattern, CFAPattern.RGGB)
        tags.set(Tag.BlackLevel, [64, 64, 64, 64])
        tags.set(Tag.WhiteLevel, [4095])
        tags.set(Tag.Make, "TestCorp")
        tags.set(Tag.Model, "TestCamera")

        # Convert to DNG
        converter = RAW2DNG()
        converter.options(tags, temp_directory, compress=False)

        with patch.object(converter, '_process') as mock_process:
            # Mock the process method to return fake DNG data
            fake_dng_data = bytearray(b'FAKE_DNG_HEADER' + b'\x00' * 1000)
            mock_process.return_value = fake_dng_data

            output_path = converter.convert(raw_data, filename="test_output")

            # Verify file was created
            assert os.path.exists(output_path)
            assert output_path.endswith("test_output.dng")

            # Verify file contents
            with open(output_path, 'rb') as f:
                file_data = f.read()
                assert file_data == fake_dng_data

    @pytest.mark.integration
    def test_picamera2_to_dng_workflow(self, temp_directory):
        """Test complete workflow from PiCamera2 to DNG."""
        # Create PiCamera2 format and metadata
        fmt = {
            "format": "SRGGB12",
            "size": (320, 240)
        }
        metadata = {
            "SensorBlackLevels": [64, 64, 64, 64],
            "ColourGains": [1.2, 1.8],
            "ColourCorrectionMatrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "ExposureTime": 16667,
            "AnalogueGain": 2.0,
            "DigitalGain": 1.0,
            "SensorTimestamp": 123456789
        }

        # Create camera and get tags
        camera = Picamera2Camera(fmt, metadata)

        # Create test image data
        raw_data = np.random.randint(0, 4095, (240, 320), dtype=np.uint16)

        # Convert to DNG
        converter = RAW2DNG()
        converter.options(camera.tags, temp_directory, compress=False)

        with patch.object(converter, '_process') as mock_process:
            fake_dng_data = bytearray(b'PICAMERA2_DNG' + b'\x00' * 2000)
            mock_process.return_value = fake_dng_data

            output_path = converter.convert(raw_data, filename="picamera2_test")

            assert os.path.exists(output_path)
            assert "picamera2_test.dng" in output_path

    @pytest.mark.integration
    def test_different_bit_depths_workflow(self, temp_directory):
        """Test workflow with different bit depths."""
        bit_depths = [8, 10, 12, 14, 16]

        for bpp in bit_depths:
            with patch('pidng.core.DNG') as mock_dng:
                mock_dng_instance = MagicMock()
                mock_dng.return_value = mock_dng_instance
                mock_dng_instance.ImageDataStrips = []
                mock_dng_instance.dataLen.return_value = 1000
                mock_dng_instance.StripOffsets = {0: 100}
                mock_dng_instance.IFDs = []

                with patch('pidng.core.dngIFD') as mock_dng_ifd:
                    mock_ifd_instance = MagicMock()
                    mock_dng_ifd.return_value = mock_ifd_instance
                    mock_ifd_instance.tags = []

                    # Create test data
                    max_val = (1 << bpp) - 1
                    raw_data = np.random.randint(0, max_val, (32, 32), dtype=np.uint16)

                    # Create tags
                    tags = DNGTags()
                    tags.set(Tag.ImageWidth, 32)
                    tags.set(Tag.ImageLength, 32)
                    tags.set(Tag.BitsPerSample, bpp)
                    tags.set(Tag.WhiteLevel, [max_val])
                    tags.set(Tag.Make, "TestCorp")
                    tags.set(Tag.Model, f"TestCamera-{bpp}bit")

                    # Convert
                    converter = RAW2DNG()
                    converter.options(tags, temp_directory, compress=False)

                    result = converter.convert(raw_data, filename=f"test_{bpp}bit")
                    assert result.endswith(f"test_{bpp}bit.dng")

    @pytest.mark.integration
    def test_float32_workflow(self, temp_directory):
        """Test workflow with float32 data."""
        # Create float32 test data
        raw_data = np.random.rand(24, 32).astype(np.float32)

        # Create tags for float data
        tags = DNGTags()
        tags.set(Tag.ImageWidth, 32)
        tags.set(Tag.ImageLength, 24)
        tags.set(Tag.BitsPerSample, 32)
        tags.set(Tag.Make, "TestCorp")
        tags.set(Tag.Model, "TestCamera-Float")

        with patch('pidng.core.DNG') as mock_dng:
            mock_dng_instance = MagicMock()
            mock_dng.return_value = mock_dng_instance
            mock_dng_instance.ImageDataStrips = []
            mock_dng_instance.dataLen.return_value = 2000
            mock_dng_instance.StripOffsets = {0: 200}
            mock_dng_instance.IFDs = []

            with patch('pidng.core.dngIFD') as mock_dng_ifd:
                mock_ifd_instance = MagicMock()
                mock_dng_ifd.return_value = mock_dng_instance
                mock_ifd_instance.tags = []

                converter = RAW2DNG()
                converter.options(tags, temp_directory, compress=False)

                result = converter.convert(raw_data, filename="test_float32")
                assert result.endswith("test_float32.dng")

    @pytest.mark.integration
    def test_buffer_output_workflow(self):
        """Test workflow returning buffer instead of file."""
        # Create test data
        raw_data = np.random.randint(0, 1023, (16, 16), dtype=np.uint16)

        # Create minimal tags
        tags = DNGTags()
        tags.set(Tag.ImageWidth, 16)
        tags.set(Tag.ImageLength, 16)
        tags.set(Tag.BitsPerSample, 10)
        tags.set(Tag.Make, "TestCorp")
        tags.set(Tag.Model, "TestCamera")

        with patch('pidng.core.DNG') as mock_dng:
            mock_dng_instance = MagicMock()
            mock_dng.return_value = mock_dng_instance
            mock_dng_instance.ImageDataStrips = []
            mock_dng_instance.dataLen.return_value = 500
            mock_dng_instance.StripOffsets = {0: 50}
            mock_dng_instance.IFDs = []

            with patch('pidng.core.dngIFD'):
                converter = RAW2DNG()
                converter.options(tags, "/tmp", compress=False)

                # Convert without filename to get buffer
                result = converter.convert(raw_data)

                assert isinstance(result, bytearray)

    @pytest.mark.integration
    def test_error_handling_workflow(self):
        """Test error handling in complete workflow."""
        converter = RAW2DNG()
        raw_data = np.random.randint(0, 1023, (16, 16), dtype=np.uint16)

        # Test convert without options
        with pytest.raises(Exception, match="Options have not been set"):
            converter.convert(raw_data)

        # Test with invalid data type
        invalid_data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        tags = DNGTags()
        tags.set(Tag.ImageWidth, 2)
        tags.set(Tag.ImageLength, 2)
        tags.set(Tag.BitsPerSample, 8)

        converter.options(tags, "/tmp", compress=False)

        with pytest.raises(Exception, match="RAW Data is not in correct format"):
            converter.convert(invalid_data)

    @pytest.mark.integration
    def test_large_image_workflow(self, temp_directory):
        """Test workflow with larger image sizes."""
        # Test with a larger but still manageable image size
        width, height = 256, 192
        raw_data = np.random.randint(0, 65535, (height, width), dtype=np.uint16)

        tags = DNGTags()
        tags.set(Tag.ImageWidth, width)
        tags.set(Tag.ImageLength, height)
        tags.set(Tag.BitsPerSample, 16)
        tags.set(Tag.Make, "TestCorp")
        tags.set(Tag.Model, "TestCamera-Large")

        with patch('pidng.core.DNG') as mock_dng:
            mock_dng_instance = MagicMock()
            mock_dng.return_value = mock_dng_instance
            mock_dng_instance.ImageDataStrips = []
            mock_dng_instance.dataLen.return_value = width * height * 2 + 1000
            mock_dng_instance.StripOffsets = {0: 1000}
            mock_dng_instance.IFDs = []

            with patch('pidng.core.dngIFD'):
                converter = RAW2DNG()
                converter.options(tags, temp_directory, compress=False)

                result = converter.convert(raw_data, filename="large_test")
                assert result.endswith("large_test.dng")


class TestCrossModuleIntegration:
    """Integration tests across multiple modules."""

    @pytest.mark.integration
    def test_packing_with_dng_creation(self, temp_directory):
        """Test that bit packing integrates correctly with DNG creation."""
        from pidng.packing import pack10, pack12, pack14

        # Test 10-bit packing integration
        raw_10bit = np.random.randint(0, 1023, (32, 32), dtype=np.uint16)

        tags = DNGTags()
        tags.set(Tag.ImageWidth, 32)
        tags.set(Tag.ImageLength, 32)
        tags.set(Tag.BitsPerSample, 10)
        tags.set(Tag.Make, "TestCorp")
        tags.set(Tag.Model, "TestCamera-10bit")

        with patch('pidng.core.pack10') as mock_pack10:
            mock_pack10.return_value = np.zeros((32, 40), dtype=np.uint8)  # 32 * 1.25

            with patch('pidng.core.DNG') as mock_dng:
                mock_dng_instance = MagicMock()
                mock_dng.return_value = mock_dng_instance
                mock_dng_instance.ImageDataStrips = []
                mock_dng_instance.dataLen.return_value = 2000
                mock_dng_instance.StripOffsets = {0: 200}
                mock_dng_instance.IFDs = []

                with patch('pidng.core.dngIFD'):
                    converter = RAW2DNG()
                    converter.options(tags, temp_directory, compress=False)

                    result = converter.convert(raw_10bit, filename="packed_10bit")

                    # Verify pack10 was called
                    mock_pack10.assert_called_once()
                    assert result.endswith("packed_10bit.dng")

    @pytest.mark.integration
    def test_camera_defs_with_dng_creation(self, temp_directory):
        """Test that camera definitions integrate correctly with DNG creation."""
        # Create a PiCamera2 camera configuration
        fmt = {"format": "SRGGB10", "size": (64, 48)}
        metadata = {
            "SensorBlackLevels": [32, 32, 32, 32],
            "ColourGains": [1.0, 1.0],
            "ColourCorrectionMatrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "ExposureTime": 33333,
            "AnalogueGain": 1.0,
            "DigitalGain": 1.0,
            "SensorTimestamp": 987654321
        }

        camera = Picamera2Camera(fmt, metadata)
        raw_data = np.random.randint(0, 1023, (48, 64), dtype=np.uint16)

        # Verify camera tags work with DNG creation
        converter = RAW2DNG()
        converter.options(camera.tags, temp_directory, compress=False)

        with patch.object(converter, '_process') as mock_process:
            mock_process.return_value = bytearray(b'CAMERA_DNG_DATA' + b'\x00' * 1500)

            result = converter.convert(raw_data, filename="camera_integration")
            assert result.endswith("camera_integration.dng")

            # Verify the camera tags were used in processing
            mock_process.assert_called_once()
            call_args = mock_process.call_args[0]
            assert np.array_equal(call_args[0], raw_data)
            assert call_args[1] == camera.tags

    @pytest.mark.integration
    def test_all_modules_working_together(self, temp_directory):
        """Test that all modules work together in a realistic scenario."""
        # This test combines:
        # - Camera definitions (camdefs)
        # - Tag management (dng)
        # - Bit packing (packing)
        # - Core DNG creation (core)
        # - All constants (defs)

        # Create realistic camera setup
        fmt = {"format": "SRGGB12", "size": (128, 96)}
        metadata = {
            "SensorBlackLevels": [64, 64, 64, 64],
            "ColourGains": [1.3, 1.7],
            "ColourCorrectionMatrix": [
                1.2, -0.2, 0.0,
                -0.1, 1.1, 0.0,
                0.0, -0.3, 1.3
            ],
            "ExposureTime": 8333,  # 1/120 second
            "AnalogueGain": 4.0,
            "DigitalGain": 2.0,
            "SensorTimestamp": 1640995200000  # Jan 1, 2022
        }

        # Create camera and verify all expected tags are set
        camera = Picamera2Camera(fmt, metadata, "Integration Test Camera")

        # Verify camera created all necessary tags
        essential_tags = [
            Tag.ImageWidth, Tag.ImageLength, Tag.BitsPerSample,
            Tag.Make, Tag.Model, Tag.CFAPattern, Tag.PhotometricInterpretation
        ]

        for tag in essential_tags:
            assert camera.tags.get(tag) is not None, f"Camera missing essential tag: {tag}"

        # Create realistic test data
        raw_data = np.random.randint(64, 3000, (96, 128), dtype=np.uint16)

        # Test the complete pipeline
        converter = RAW2DNG()
        converter.options(camera.tags, temp_directory, compress=False)

        with patch('pidng.core.pack12') as mock_pack12:
            # Mock pack12 to return appropriately sized data
            mock_pack12.return_value = np.zeros((96, 192), dtype=np.uint8)  # 128 * 1.5

            with patch('pidng.core.DNG') as mock_dng:
                mock_dng_instance = MagicMock()
                mock_dng.return_value = mock_dng_instance
                mock_dng_instance.ImageDataStrips = []
                mock_dng_instance.dataLen.return_value = 25000
                mock_dng_instance.StripOffsets = {0: 1000}
                mock_dng_instance.IFDs = []

                with patch('pidng.core.dngIFD'):
                    result = converter.convert(raw_data, filename="full_integration_test")

                    # Verify complete workflow succeeded
                    assert result.endswith("full_integration_test.dng")
                    assert os.path.exists(result)

                    # Verify bit packing was called for 12-bit data
                    mock_pack12.assert_called_once()
