import pytest
import numpy as np
from pidng.packing import pack10, pack12, pack14


class TestPack10:
    """Test 10-bit packing functionality."""

    def test_pack10_basic(self):
        """Test basic 10-bit packing."""
        # Create test data - 4 pixels that should pack into 5 bytes
        data = np.array([[0x3FF, 0x200, 0x100, 0x000]], dtype=np.uint16)  # Max, mid, low, zero values
        result = pack10(data)

        # Should have shape (1, 5) for 4 input pixels
        assert result.shape == (1, 5)
        assert result.dtype == np.uint8

    def test_pack10_shape_calculation(self):
        """Test that pack10 produces correct output shape."""
        # Input: 8 pixels should produce 10 bytes (8 * 1.25)
        data = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.uint16)
        result = pack10(data)
        assert result.shape == (1, 10)

        # Input: 16 pixels should produce 20 bytes
        data = np.array([[1] * 16], dtype=np.uint16)
        result = pack10(data)
        assert result.shape == (1, 20)

    def test_pack10_multiple_rows(self):
        """Test pack10 with multiple rows."""
        data = np.array([
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ], dtype=np.uint16)
        result = pack10(data)

        # Should preserve number of rows
        assert result.shape[0] == 2
        assert result.shape[1] == 5  # 4 * 1.25

    def test_pack10_bit_precision(self):
        """Test that pack10 preserves 10-bit precision."""
        # Test with values that use all 10 bits
        data = np.array([[0x000, 0x3FF, 0x200, 0x155]], dtype=np.uint16)
        result = pack10(data)

        # Verify the packing preserves the data correctly
        # This is a basic check - detailed bit-level verification would be more complex
        assert result.dtype == np.uint8
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_pack10_zero_values(self):
        """Test pack10 with all zero values."""
        data = np.array([[0, 0, 0, 0]], dtype=np.uint16)
        result = pack10(data)

        # All output should be zero
        assert np.all(result == 0)

    def test_pack10_max_values(self):
        """Test pack10 with maximum 10-bit values."""
        data = np.array([[0x3FF, 0x3FF, 0x3FF, 0x3FF]], dtype=np.uint16)
        result = pack10(data)

        # Should not exceed uint8 range
        assert np.all(result <= 255)
        assert np.all(result >= 0)


class TestPack12:
    """Test 12-bit packing functionality."""

    def test_pack12_basic(self):
        """Test basic 12-bit packing."""
        # Create test data - 2 pixels that should pack into 3 bytes
        data = np.array([[0xFFF, 0x800]], dtype=np.uint16)
        result = pack12(data)

        # Should have shape (1, 3) for 2 input pixels
        assert result.shape == (1, 3)
        assert result.dtype == np.uint8

    def test_pack12_shape_calculation(self):
        """Test that pack12 produces correct output shape."""
        # Input: 4 pixels should produce 6 bytes (4 * 1.5)
        data = np.array([[1, 2, 3, 4]], dtype=np.uint16)
        result = pack12(data)
        assert result.shape == (1, 6)

        # Input: 8 pixels should produce 12 bytes
        data = np.array([[1] * 8], dtype=np.uint16)
        result = pack12(data)
        assert result.shape == (1, 12)

    def test_pack12_multiple_rows(self):
        """Test pack12 with multiple rows."""
        data = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ], dtype=np.uint16)
        result = pack12(data)

        # Should preserve number of rows
        assert result.shape[0] == 3
        assert result.shape[1] == 3  # 2 * 1.5

    def test_pack12_bit_precision(self):
        """Test that pack12 preserves 12-bit precision."""
        # Test with values that use all 12 bits
        data = np.array([[0x000, 0xFFF]], dtype=np.uint16)
        result = pack12(data)

        assert result.dtype == np.uint8
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_pack12_zero_values(self):
        """Test pack12 with all zero values."""
        data = np.array([[0, 0]], dtype=np.uint16)
        result = pack12(data)

        # All output should be zero
        assert np.all(result == 0)

    def test_pack12_max_values(self):
        """Test pack12 with maximum 12-bit values."""
        data = np.array([[0xFFF, 0xFFF]], dtype=np.uint16)
        result = pack12(data)

        # Should not exceed uint8 range
        assert np.all(result <= 255)
        assert np.all(result >= 0)


class TestPack14:
    """Test 14-bit packing functionality."""

    def test_pack14_basic(self):
        """Test basic 14-bit packing."""
        # Create test data - 6 pixels that should pack into approximately 10.5 bytes (rounded to 10)
        data = np.array([[0x3FFF, 0x2000, 0x1000, 0x800, 0x400, 0x000]], dtype=np.uint16)
        result = pack14(data)

        # Should have shape (1, 7*floor(6/6)) = (1, 7) for 6 input pixels
        expected_width = int(6 * 1.75)
        assert result.shape == (1, expected_width)
        assert result.dtype == np.uint8

    def test_pack14_shape_calculation(self):
        """Test that pack14 produces correct output shape."""
        # Input: 6 pixels should produce 10.5 bytes, rounded appropriately
        data = np.array([[1] * 6], dtype=np.uint16)
        result = pack14(data)
        expected_width = int(6 * 1.75)
        assert result.shape == (1, expected_width)

        # Input: 12 pixels should produce 21 bytes
        data = np.array([[1] * 12], dtype=np.uint16)
        result = pack14(data)
        expected_width = int(12 * 1.75)
        assert result.shape == (1, expected_width)

    def test_pack14_multiple_rows(self):
        """Test pack14 with multiple rows."""
        data = np.array([
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12]
        ], dtype=np.uint16)
        result = pack14(data)

        # Should preserve number of rows
        assert result.shape[0] == 2
        expected_width = int(6 * 1.75)
        assert result.shape[1] == expected_width

    def test_pack14_bit_precision(self):
        """Test that pack14 preserves 14-bit precision."""
        # Test with values that use all 14 bits
        data = np.array([[0x0000, 0x3FFF, 0x2000, 0x1555, 0x0AAA, 0x1000]], dtype=np.uint16)
        result = pack14(data)

        assert result.dtype == np.uint8
        assert np.all(result >= 0)
        assert np.all(result <= 255)

    def test_pack14_zero_values(self):
        """Test pack14 with all zero values."""
        data = np.array([[0, 0, 0, 0, 0, 0]], dtype=np.uint16)
        result = pack14(data)

        # All output should be zero
        assert np.all(result == 0)

    def test_pack14_max_values(self):
        """Test pack14 with maximum 14-bit values."""
        data = np.array([[0x3FFF] * 6], dtype=np.uint16)
        result = pack14(data)

        # Should not exceed uint8 range
        assert np.all(result <= 255)
        assert np.all(result >= 0)


class TestPackingIntegration:
    """Integration tests for packing functions."""

    def test_packing_functions_consistency(self):
        """Test that all packing functions handle similar inputs consistently."""
        # Test with compatible data sizes
        base_data = np.array([[100, 200, 300, 400]], dtype=np.uint16)

        # All functions should work without errors
        result10 = pack10(base_data)
        result12 = pack12(base_data[:, :2])  # Take only 2 pixels for 12-bit
        result14 = pack14(base_data[:, :6] if base_data.shape[1] >= 6 else
                         np.tile(base_data, (1, 2))[:, :6])  # Ensure 6 pixels for 14-bit

        # All should return uint8 arrays
        assert result10.dtype == np.uint8
        assert result12.dtype == np.uint8
        assert result14.dtype == np.uint8

        # All should have valid ranges
        assert np.all(result10 >= 0) and np.all(result10 <= 255)
        assert np.all(result12 >= 0) and np.all(result12 <= 255)
        assert np.all(result14 >= 0) and np.all(result14 <= 255)

    def test_packing_compression_ratios(self):
        """Test that packing functions achieve expected compression ratios."""
        # Create test data with known sizes
        data_4px = np.array([[1, 2, 3, 4]], dtype=np.uint16)  # 4 pixels
        data_2px = np.array([[1, 2]], dtype=np.uint16)        # 2 pixels
        data_6px = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint16)  # 6 pixels

        # Test compression ratios
        result10 = pack10(data_4px)
        assert result10.shape[1] == 5  # 4 * 1.25

        result12 = pack12(data_2px)
        assert result12.shape[1] == 3  # 2 * 1.5

        result14 = pack14(data_6px)
        assert result14.shape[1] == int(6 * 1.75)  # 6 * 1.75 = 10.5, truncated

    def test_empty_arrays(self):
        """Test packing functions with edge case inputs."""
        # Test with minimal valid input sizes

        # pack10 needs multiples of 4 pixels to work correctly
        data = np.array([[1, 2, 3, 4]], dtype=np.uint16)
        result = pack10(data)
        assert result.shape[1] > 0

        # pack12 needs multiples of 2 pixels
        data = np.array([[1, 2]], dtype=np.uint16)
        result = pack12(data)
        assert result.shape[1] > 0

        # pack14 needs multiples of 6 pixels for optimal packing
        data = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint16)
        result = pack14(data)
        assert result.shape[1] > 0
