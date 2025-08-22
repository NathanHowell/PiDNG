import pytest
from pidng.defs import (
    Compression, PreviewColorSpace, Orientation, DNGVersion,
    PhotometricInterpretation, CFAPattern, CalibrationIlluminant
)


class TestCompression:
    """Test Compression constants."""

    def test_compression_values(self):
        """Test that compression constants have correct values."""
        assert Compression.Uncompressed == 1
        assert Compression.LJ92 == 7
        assert Compression.Lossy_JPEG == 34892


class TestPreviewColorSpace:
    """Test PreviewColorSpace constants."""

    def test_preview_color_space_values(self):
        """Test that preview color space constants have correct values."""
        assert PreviewColorSpace.Unknown == 0
        assert PreviewColorSpace.Gray_Gamma_22 == 1
        assert PreviewColorSpace.sRGB == 2
        assert PreviewColorSpace.Adobe_RGB == 3
        assert PreviewColorSpace.ProPhoto_RGB == 4


class TestOrientation:
    """Test Orientation constants."""

    def test_orientation_values(self):
        """Test that orientation constants have correct values."""
        assert Orientation.Horizontal == 1
        assert Orientation.MirrorH == 2
        assert Orientation.Rotate180 == 3
        assert Orientation.MirrorV == 4


class TestDNGVersion:
    """Test DNGVersion constants."""

    def test_dng_version_values(self):
        """Test that DNG version constants have correct values."""
        assert DNGVersion.V1_0 == [1, 0, 0, 0]
        assert DNGVersion.V1_1 == [1, 1, 0, 0]
        assert DNGVersion.V1_2 == [1, 2, 0, 0]
        assert DNGVersion.V1_3 == [1, 3, 0, 0]
        assert DNGVersion.V1_4 == [1, 4, 0, 0]
        assert DNGVersion.V1_5 == [1, 5, 0, 0]
        assert DNGVersion.V1_6 == [1, 6, 0, 0]

    def test_dng_version_progression(self):
        """Test that DNG versions follow logical progression."""
        versions = [
            DNGVersion.V1_0, DNGVersion.V1_1, DNGVersion.V1_2,
            DNGVersion.V1_3, DNGVersion.V1_4, DNGVersion.V1_5, DNGVersion.V1_6
        ]

        # Each version should increment the minor version
        for i in range(len(versions) - 1):
            current = versions[i]
            next_version = versions[i + 1]
            assert current[0] == next_version[0]  # Major version stays same
            assert current[1] + 1 == next_version[1]  # Minor version increments
            assert current[2] == next_version[2] == 0  # Patch version stays 0
            assert current[3] == next_version[3] == 0  # Build version stays 0


class TestPhotometricInterpretation:
    """Test PhotometricInterpretation constants."""

    def test_photometric_interpretation_values(self):
        """Test that photometric interpretation constants have correct values."""
        assert PhotometricInterpretation.WhiteIsZero == 0
        assert PhotometricInterpretation.BlackIsZero == 1
        assert PhotometricInterpretation.RGB == 2
        assert PhotometricInterpretation.Linear_Raw == 34892
        assert PhotometricInterpretation.Color_Filter_Array == 32803


class TestCFAPattern:
    """Test CFAPattern constants."""

    def test_cfa_pattern_values(self):
        """Test that CFA pattern constants have correct values."""
        assert CFAPattern.BGGR == [2, 1, 1, 0]
        assert CFAPattern.GBRG == [1, 2, 0, 1]
        assert CFAPattern.GRBG == [1, 0, 2, 1]
        assert CFAPattern.RGGB == [0, 1, 1, 2]

    def test_cfa_pattern_structure(self):
        """Test that CFA patterns have correct structure."""
        patterns = [CFAPattern.BGGR, CFAPattern.GBRG, CFAPattern.GRBG, CFAPattern.RGGB]

        for pattern in patterns:
            # Each pattern should be a 2x2 arrangement (4 elements)
            assert len(pattern) == 4
            # Should contain values 0, 1, 2 representing R, G, B
            assert all(val in [0, 1, 2] for val in pattern)
            # Should have exactly 2 green pixels (value 1)
            assert pattern.count(1) == 2
            # Should have exactly 1 red pixel (value 0)
            assert pattern.count(0) == 1
            # Should have exactly 1 blue pixel (value 2)
            assert pattern.count(2) == 1

    def test_cfa_pattern_differences(self):
        """Test that different CFA patterns are actually different."""
        patterns = {
            'BGGR': CFAPattern.BGGR,
            'GBRG': CFAPattern.GBRG,
            'GRBG': CFAPattern.GRBG,
            'RGGB': CFAPattern.RGGB
        }

        # All patterns should be different from each other
        pattern_values = list(patterns.values())
        for i in range(len(pattern_values)):
            for j in range(i + 1, len(pattern_values)):
                assert pattern_values[i] != pattern_values[j]


class TestCalibrationIlluminant:
    """Test CalibrationIlluminant constants."""

    def test_calibration_illuminant_values(self):
        """Test that calibration illuminant constants have correct values."""
        assert CalibrationIlluminant.Unknown == 0
        assert CalibrationIlluminant.Daylight == 1
        assert CalibrationIlluminant.Fluorescent == 2
        assert CalibrationIlluminant.Tungsten_Incandescent == 3
        assert CalibrationIlluminant.Flash == 4


class TestDefinitionsIntegration:
    """Integration tests for definitions module."""

    def test_all_constants_are_immutable_types(self):
        """Test that all constants use immutable types."""
        # Test compression constants
        assert isinstance(Compression.Uncompressed, int)
        assert isinstance(Compression.LJ92, int)
        assert isinstance(Compression.Lossy_JPEG, int)

        # Test orientation constants
        assert isinstance(Orientation.Horizontal, int)
        assert isinstance(Orientation.MirrorH, int)

        # Test DNG version constants (should be lists, but that's the design)
        assert isinstance(DNGVersion.V1_4, list)
        assert len(DNGVersion.V1_4) == 4

        # Test CFA patterns (should be lists)
        assert isinstance(CFAPattern.RGGB, list)
        assert len(CFAPattern.RGGB) == 4

    def test_constants_for_typical_use_case(self):
        """Test constants that would be used in a typical DNG creation workflow."""
        # These are the most commonly used constants
        common_constants = {
            'compression': Compression.LJ92,
            'orientation': Orientation.Horizontal,
            'dng_version': DNGVersion.V1_4,
            'photometric': PhotometricInterpretation.Color_Filter_Array,
            'cfa_pattern': CFAPattern.RGGB,
            'illuminant': CalibrationIlluminant.Daylight,
            'color_space': PreviewColorSpace.sRGB
        }

        # All should be valid values
        assert all(val is not None for val in common_constants.values())

        # Test specific expected values for common workflow
        assert common_constants['compression'] == 7  # LJ92
        assert common_constants['orientation'] == 1   # Horizontal
        assert common_constants['dng_version'] == [1, 4, 0, 0]  # V1.4
        assert common_constants['photometric'] == 32803  # CFA
        assert common_constants['cfa_pattern'] == [0, 1, 1, 2]  # RGGB
        assert common_constants['illuminant'] == 1  # Daylight
        assert common_constants['color_space'] == 2  # sRGB

    def test_version_compatibility(self):
        """Test version compatibility relationships."""
        # V1_4 should be newer than V1_2
        v12 = DNGVersion.V1_2
        v14 = DNGVersion.V1_4

        assert v12[1] < v14[1]  # Minor version comparison

        # Latest version should be V1_6
        latest = DNGVersion.V1_6
        assert latest == [1, 6, 0, 0]

    def test_cfa_pattern_naming_consistency(self):
        """Test that CFA pattern names match their actual arrangements."""
        # RGGB means: Red-Green / Green-Blue in a 2x2 grid
        # Pattern [0,1,1,2] means [R,G,G,B] reading left-to-right, top-to-bottom
        rggb = CFAPattern.RGGB
        assert rggb[0] == 0  # Top-left: Red
        assert rggb[1] == 1  # Top-right: Green
        assert rggb[2] == 1  # Bottom-left: Green
        assert rggb[3] == 2  # Bottom-right: Blue

        # BGGR means: Blue-Green / Green-Red
        bggr = CFAPattern.BGGR
        assert bggr[0] == 2  # Top-left: Blue
        assert bggr[1] == 1  # Top-right: Green
        assert bggr[2] == 1  # Bottom-left: Green
        assert bggr[3] == 0  # Bottom-right: Red
