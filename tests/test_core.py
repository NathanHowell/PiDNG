import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from pidng.core import DNGBASE, RAW2DNG, DNGTags
from pidng.dng import Tag
from pidng.defs import Compression, DNGVersion, SampleFormat, Orientation, PhotometricInterpretation, CFAPattern


class TestDNGBASE:
    def setup_method(self):
        self.dng_base = DNGBASE()

    def test_init(self):
        """Test DNGBASE initialization."""
        assert self.dng_base.compress is None
        assert self.dng_base.path is None
        assert self.dng_base.tags is None
        assert self.dng_base.filter is None

    def test_data_condition_valid_uint16(self):
        """Test data condition validation with valid uint16 data."""
        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        # Should not raise exception
        self.dng_base._data_condition(data)

    def test_data_condition_valid_float32(self):
        """Test data condition validation with valid float32 data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # Should not raise exception
        self.dng_base._data_condition(data)

    def test_data_condition_invalid_dtype(self):
        """Test data condition validation with invalid data type."""
        data = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        with pytest.raises(Exception, match="RAW Data is not in correct format"):
            self.dng_base._data_condition(data)

    def test_tags_condition_missing_width(self):
        """Test tags condition validation with missing width."""
        tags = DNGTags()
        tags.set(Tag.ImageLength, 100)
        tags.set(Tag.BitsPerSample, 16)

        with pytest.raises(Exception, match="No width is defined in tags"):
            self.dng_base._tags_condition(tags)

    def test_tags_condition_missing_height(self):
        """Test tags condition validation with missing height."""
        tags = DNGTags()
        tags.set(Tag.ImageWidth, 100)
        tags.set(Tag.BitsPerSample, 16)

        with pytest.raises(Exception, match="No height is defined in tags"):
            self.dng_base._tags_condition(tags)

    def test_tags_condition_missing_bits_per_sample(self):
        """Test tags condition validation with missing bits per sample."""
        tags = DNGTags()
        tags.set(Tag.ImageWidth, 100)
        tags.set(Tag.ImageLength, 100)

        with pytest.raises(Exception, match="Bit per pixel is not defined"):
            self.dng_base._tags_condition(tags)

    def test_tags_condition_valid(self):
        """Test tags condition validation with valid tags."""
        tags = DNGTags()
        tags.set(Tag.ImageWidth, 100)
        tags.set(Tag.ImageLength, 100)
        tags.set(Tag.BitsPerSample, 16)

        # Should not raise exception
        self.dng_base._tags_condition(tags)

    def test_unpack_pixels_default(self):
        """Test default pixel unpacking (no-op)."""
        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        result = self.dng_base._unpack_pixels(data)
        np.testing.assert_array_equal(result, data)

    def test_filter_no_filter(self):
        """Test filtering with no filter applied."""
        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
        result = self.dng_base._filter(data, None)
        np.testing.assert_array_equal(result, data)

    def test_filter_with_valid_filter(self):
        """Test filtering with a valid filter function."""
        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        def test_filter(img):
            return img * 2

        result = self.dng_base._filter(data, test_filter)
        expected = data * 2
        np.testing.assert_array_equal(result, expected)

    def test_filter_invalid_return_type(self):
        """Test filtering with filter that returns invalid type."""
        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        def bad_filter(img):
            return "not an array"

        with pytest.raises(TypeError, match="return value is not a valid numpy array"):
            self.dng_base._filter(data, bad_filter)

    def test_filter_invalid_shape(self):
        """Test filtering with filter that returns wrong shape."""
        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        def bad_filter(img):
            return np.array([1, 2, 3], dtype=np.uint16)

        with pytest.raises(ValueError, match="return array does not have the same shape"):
            self.dng_base._filter(data, bad_filter)

    def test_filter_invalid_dtype(self):
        """Test filtering with filter that returns wrong dtype."""
        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        def bad_filter(img):
            return img.astype(np.uint8)

        with pytest.raises(ValueError, match="array data type is invalid"):
            self.dng_base._filter(data, bad_filter)

    def test_options_valid(self):
        """Test setting options with valid parameters."""
        tags = DNGTags()
        tags.set(Tag.ImageWidth, 100)
        tags.set(Tag.ImageLength, 100)
        tags.set(Tag.BitsPerSample, 16)

        self.dng_base.options(tags, "/test/path", compress=True)

        assert self.dng_base.tags == tags
        assert self.dng_base.path == "/test/path"
        assert self.dng_base.compress is True

    def test_convert_no_options(self):
        """Test convert without setting options first."""
        data = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        with pytest.raises(Exception, match="Options have not been set"):
            self.dng_base.convert(data)


class TestRAW2DNG:
    def setup_method(self):
        self.raw2dng = RAW2DNG()

    def test_inheritance(self):
        """Test that RAW2DNG inherits from DNGBASE."""
        assert isinstance(self.raw2dng, DNGBASE)

    def create_valid_tags(self):
        """Helper to create valid DNG tags."""
        tags = DNGTags()
        tags.set(Tag.ImageWidth, 4)
        tags.set(Tag.ImageLength, 3)
        tags.set(Tag.BitsPerSample, 16)
        tags.set(Tag.Orientation, Orientation.Horizontal)
        tags.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
        tags.set(Tag.SamplesPerPixel, 1)
        tags.set(Tag.CFARepeatPatternDim, [2, 2])
        tags.set(Tag.CFAPattern, CFAPattern.GBRG)
        return tags

    @patch('pidng.core.DNG')
    @patch('pidng.core.dngIFD')
    @patch('pidng.core.dngTag')
    def test_process_uncompressed_16bit(self, mock_dng_tag, mock_dng_ifd, mock_dng):
        """Test processing uncompressed 16-bit data."""
        # Setup mock objects
        mock_dng_instance = MagicMock()
        mock_dng.return_value = mock_dng_instance
        mock_dng_instance.ImageDataStrips = []
        mock_dng_instance.dataLen.return_value = 1000
        mock_dng_instance.StripOffsets = {0: 100}
        mock_dng_instance.IFDs = []

        mock_ifd_instance = MagicMock()
        mock_dng_ifd.return_value = mock_ifd_instance
        mock_ifd_instance.tags = []

        # Create test data
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint16)
        tags = self.create_valid_tags()

        # Call the method
        result = self.raw2dng._process(data, tags, compress=False)

        # Verify result
        assert isinstance(result, bytearray)
        assert len(result) == 1000

    def test_convert_with_filename(self):
        """Test convert method with filename output."""
        tags = self.create_valid_tags()
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint16)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.raw2dng.options(tags, temp_dir, compress=False)

            with patch.object(self.raw2dng, '_process') as mock_process:
                mock_process.return_value = bytearray(b'fake_dng_data')

                result = self.raw2dng.convert(data, filename="test")

                assert result.endswith("test.dng")
                assert os.path.exists(result)

    def test_convert_without_filename(self):
        """Test convert method returning buffer."""
        tags = self.create_valid_tags()
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint16)

        self.raw2dng.options(tags, "/test/path", compress=False)

        with patch.object(self.raw2dng, '_process') as mock_process:
            mock_process.return_value = bytearray(b'fake_dng_data')

            result = self.raw2dng.convert(data)

            assert isinstance(result, bytearray)
            assert result == bytearray(b'fake_dng_data')

    def test_convert_adds_dng_extension(self):
        """Test that convert method adds .dng extension if missing."""
        tags = self.create_valid_tags()
        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.uint16)

        with tempfile.TemporaryDirectory() as temp_dir:
            self.raw2dng.options(tags, temp_dir, compress=False)

            with patch.object(self.raw2dng, '_process') as mock_process:
                mock_process.return_value = bytearray(b'fake_dng_data')

                result = self.raw2dng.convert(data, filename="test_no_ext")

                assert result.endswith("test_no_ext.dng")

    def test_process_float32_data(self):
        """Test processing float32 data."""
        tags = self.create_valid_tags()
        data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)

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

                result = self.raw2dng._process(data, tags, compress=False)
                assert isinstance(result, bytearray)

    def test_process_float32_compression_error(self):
        """Test that float32 data with compression raises error."""
        tags = self.create_valid_tags()
        data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)

        with pytest.raises(Exception, match="Compression is not supported for floating-point data"):
            self.raw2dng._process(data, tags, compress=True)
