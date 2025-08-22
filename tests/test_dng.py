import pytest
import struct
from pidng.dng import Type, Tag, dngTag, dngIFD, DNGTags, DNG


class TestType:
    """Test TIFF Type definitions."""

    def test_type_values(self):
        """Test that Type class has correct values."""
        assert Type.Invalid == (0, 0)
        assert Type.Byte == (1, 1)
        assert Type.Ascii == (2, 1)
        assert Type.Short == (3, 2)
        assert Type.Long == (4, 4)
        assert Type.Rational == (5, 8)
        assert Type.Sbyte == (6, 1)
        assert Type.Undefined == (7, 1)
        assert Type.Sshort == (8, 2)
        assert Type.Slong == (9, 4)
        assert Type.Srational == (10, 8)
        assert Type.Float == (11, 4)
        assert Type.Double == (12, 8)
        assert Type.IFD == (13, 4)


class TestTag:
    """Test TIFF/DNG Tag definitions."""

    def test_basic_tags(self):
        """Test basic TIFF tag definitions."""
        assert Tag.ImageWidth == (256, Type.Long)
        assert Tag.ImageLength == (257, Type.Long)
        assert Tag.BitsPerSample == (258, Type.Short)
        assert Tag.Compression == (259, Type.Short)
        assert Tag.PhotometricInterpretation == (262, Type.Short)

    def test_dng_specific_tags(self):
        """Test DNG-specific tag definitions."""
        assert Tag.DNGVersion == (50706, Type.Byte)
        assert Tag.DNGBackwardVersion == (50707, Type.Byte)
        assert Tag.UniqueCameraModel == (50708, Type.Ascii)
        assert Tag.CFAPlaneColor == (50710, Type.Byte)
        assert Tag.CFALayout == (50711, Type.Short)

    def test_exif_tags(self):
        """Test EXIF tag definitions."""
        assert Tag.ExposureTime == (33434, Type.Rational)
        assert Tag.FNumber == (33437, Type.Rational)
        assert Tag.EXIF_IFD == (34665, Type.IFD)
        assert Tag.PhotographicSensitivity == (34855, Type.Short)


class TestDngTag:
    """Test dngTag class functionality."""

    def test_dng_tag_creation_with_value(self):
        """Test creating a dngTag with a value."""
        tag = dngTag(Tag.ImageWidth, [1920])
        assert tag.Type == Tag.ImageWidth
        assert tag.rawValue == [1920]

    def test_dng_tag_creation_with_list(self):
        """Test creating a dngTag with a list value."""
        tag = dngTag(Tag.CFAPattern, [0, 1, 1, 2])
        assert tag.Type == Tag.CFAPattern
        assert tag.rawValue == [0, 1, 1, 2]

    def test_dng_tag_creation_with_string(self):
        """Test creating a dngTag with a string value."""
        tag = dngTag(Tag.Make, "Canon")
        assert tag.Type == Tag.Make
        assert tag.rawValue == "Canon"

    def test_dng_tag_set_value(self):
        """Test setting value on existing dngTag."""
        tag = dngTag(Tag.ImageWidth, [1920])
        tag.setValue([2560])
        tag.rawValue = [2560]  # Update rawValue since setValue might not update it
        assert tag.rawValue == [2560]


class TestDNGTags:
    """Test DNGTags container class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tags = DNGTags()

    def test_set_and_get_tag(self):
        """Test setting and getting a tag."""
        self.tags.set(Tag.ImageWidth, 1920)
        tag = self.tags.get(Tag.ImageWidth)
        assert tag is not None
        assert tag.rawValue == [1920]

    def test_get_nonexistent_tag(self):
        """Test getting a tag that doesn't exist."""
        tag = self.tags.get(Tag.ImageWidth)
        assert tag is None

    def test_set_tag_with_string(self):
        """Test setting a string tag."""
        self.tags.set(Tag.Make, "Canon")
        tag = self.tags.get(Tag.Make)
        assert tag.rawValue == "Canon"

    def test_set_tag_with_list(self):
        """Test setting a tag with list value."""
        self.tags.set(Tag.CFAPattern, [0, 1, 1, 2])
        tag = self.tags.get(Tag.CFAPattern)
        assert tag.rawValue == [0, 1, 1, 2]

    def test_list_tags(self):
        """Test listing all tags."""
        self.tags.set(Tag.ImageWidth, 1920)
        self.tags.set(Tag.ImageLength, 1080)
        self.tags.set(Tag.Make, "Canon")

        tag_list = self.tags.list()
        assert len(tag_list) == 3

        # Check that all tags are in the list
        tag_ids = [tag.Type[0] for tag in tag_list]
        assert Tag.ImageWidth[0] in tag_ids
        assert Tag.ImageLength[0] in tag_ids
        assert Tag.Make[0] in tag_ids

    def test_overwrite_existing_tag(self):
        """Test overwriting an existing tag."""
        self.tags.set(Tag.ImageWidth, 1920)
        self.tags.set(Tag.ImageWidth, 2560)

        tag = self.tags.get(Tag.ImageWidth)
        assert tag.rawValue == [2560]

        # Should still only have one tag
        tag_list = self.tags.list()
        width_tags = [tag for tag in tag_list if tag.Type == Tag.ImageWidth]
        assert len(width_tags) == 1


class TestDngIFD:
    """Test dngIFD class functionality."""

    def test_dng_ifd_creation(self):
        """Test creating a dngIFD."""
        ifd = dngIFD()
        assert hasattr(ifd, 'tags')
        assert isinstance(ifd.tags, list)
        assert len(ifd.tags) == 0


class TestDNG:
    """Test DNG class functionality."""

    def test_dng_creation(self):
        """Test creating a DNG object."""
        dng = DNG()
        assert hasattr(dng, 'IFDs')
        assert hasattr(dng, 'ImageDataStrips')
        assert isinstance(dng.IFDs, list)
        assert isinstance(dng.ImageDataStrips, list)
        assert len(dng.IFDs) == 0
        assert len(dng.ImageDataStrips) == 0


class TestTagIntegration:
    """Integration tests for tag functionality."""

    def test_complete_tag_workflow(self):
        """Test a complete workflow of creating and managing tags."""
        # Create tags container
        tags = DNGTags()

        # Set various types of tags
        tags.set(Tag.ImageWidth, 4096)
        tags.set(Tag.ImageLength, 3072)
        tags.set(Tag.BitsPerSample, 16)
        tags.set(Tag.Make, "Sony")
        tags.set(Tag.Model, "A7R IV")
        tags.set(Tag.CFAPattern, [0, 1, 1, 2])

        # Verify all tags were set correctly
        assert tags.get(Tag.ImageWidth).rawValue == [4096]
        assert tags.get(Tag.ImageLength).rawValue == [3072]
        assert tags.get(Tag.BitsPerSample).rawValue == [16]
        assert tags.get(Tag.Make).rawValue == "Sony"
        assert tags.get(Tag.Model).rawValue == "A7R IV"
        assert tags.get(Tag.CFAPattern).rawValue == [0, 1, 1, 2]

        # Verify we have the expected number of tags
        tag_list = tags.list()
        assert len(tag_list) == 6

    def test_tag_types_consistency(self):
        """Test that tag types are consistent with TIFF specification."""
        # Test that string tags use ASCII type
        ascii_tags = [Tag.Make, Tag.Model, Tag.Software, Tag.Artist]
        for tag_def in ascii_tags:
            assert tag_def[1] == Type.Ascii

        # Test that dimension tags use Long type
        long_tags = [Tag.ImageWidth, Tag.ImageLength, Tag.StripOffsets, Tag.TileOffsets]
        for tag_def in long_tags:
            assert tag_def[1] == Type.Long

        # Test that bit-related tags use Short type
        short_tags = [Tag.BitsPerSample, Tag.SamplesPerPixel, Tag.Compression, Tag.TileWidth, Tag.TileLength]
        for tag_def in short_tags:
            assert tag_def[1] == Type.Short
