"""I/O helpers shared across the geolocation pipeline scripts."""

import glob
import os
from typing import List


def gather_images_recursive(root_dir: str) -> List[str]:
    """Find all images under ``root_dir`` (recursive), sorted and deduplicated.

    Searches for common case-variants of jpg/jpeg/png. Returns a sorted
    list with no duplicates so deployments that mix case-sensitive and
    case-insensitive filesystems produce stable output.

    Parameters
    ----------
    root_dir : str
        Root directory to search.

    Returns
    -------
    List[str]
        Sorted, deduplicated list of image file paths.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths: List[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    return sorted(set(paths))
