"""Timestamp parsing helpers for camera-trap filename conventions."""

import datetime
import os
from typing import Optional


def parse_iso_timestamp_from_filename(path_or_name: str) -> Optional[str]:
    """Extract an ISO timestamp from a SmartWilds-style filename.

    Filenames following the pattern ``<prefix>_YYMMDDHHMMSS_<suffix>.<ext>``
    (used by the SmartWilds dataset and many trail-camera firmwares) are
    parsed. For example::

        NSCF0001_240714183022_0088.JPG  ->  "2024-07-14T18:30:22"

    Parameters
    ----------
    path_or_name : str
        Either a full path or a bare filename.

    Returns
    -------
    Optional[str]
        ISO 8601 timestamp string if successfully parsed, otherwise ``None``.
    """
    try:
        base = os.path.basename(path_or_name)
        parts = base.split("_")
        if len(parts) < 2:
            return None

        dt = parts[1]
        if len(dt) < 12 or not dt[:12].isdigit():
            return None

        return datetime.datetime(
            int("20" + dt[0:2]),  # year
            int(dt[2:4]),          # month
            int(dt[4:6]),          # day
            int(dt[6:8]),          # hour
            int(dt[8:10]),         # minute
            int(dt[10:12]),        # second
        ).isoformat()
    except (ValueError, IndexError):
        return None
