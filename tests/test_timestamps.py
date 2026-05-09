"""Unit tests for timestamps module."""

from camera_trap_geolocation.timestamps import parse_iso_timestamp_from_filename


def test_smartwilds_style_filename():
    """A SmartWilds-style filename parses cleanly."""
    fname = "NSCF0001_240714183022_0088.JPG"
    assert parse_iso_timestamp_from_filename(fname) == "2024-07-14T18:30:22"


def test_full_path_works():
    """Parsing should ignore the directory portion of the path."""
    path = "/data/site_A/NSCF0001_231031200001_0088.JPG"
    assert parse_iso_timestamp_from_filename(path) == "2023-10-31T20:00:01"


def test_no_underscore_returns_none():
    """A filename with no underscore-separated timestamp returns None."""
    assert parse_iso_timestamp_from_filename("plain_filename.jpg") is None


def test_short_timestamp_returns_none():
    """A timestamp shorter than 12 digits returns None."""
    assert parse_iso_timestamp_from_filename("PREFIX_2407_SUFFIX.JPG") is None


def test_non_digit_timestamp_returns_none():
    """If the timestamp portion is not all digits, returns None."""
    assert parse_iso_timestamp_from_filename("PREFIX_xxxxxxxxxxxx_SUFFIX.JPG") is None


def test_invalid_date_returns_none():
    """Month 13 / day 99 etc. should return None rather than raising."""
    # 991399999999 = year 2099, month 13, day 99 -> invalid datetime
    assert parse_iso_timestamp_from_filename("PREFIX_991399999999_SUFFIX.JPG") is None
