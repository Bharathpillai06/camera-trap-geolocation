"""camera-trap-geolocation: animal GPS estimation from camera trap imagery."""

from camera_trap_geolocation.geometry import (
    estimate_distance_and_angle,
    project_gps,
    focal_length_px,
    parse_camera_bearing,
)
from camera_trap_geolocation.timestamps import parse_iso_timestamp_from_filename
from camera_trap_geolocation.io_utils import gather_images_recursive

__version__ = "0.1.0"

__all__ = [
    "estimate_distance_and_angle",
    "project_gps",
    "focal_length_px",
    "parse_camera_bearing",
    "parse_iso_timestamp_from_filename",
    "gather_images_recursive",
]
