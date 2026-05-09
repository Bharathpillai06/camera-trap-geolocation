"""
Pinhole geometry and GPS forward projection for camera-trap detections.

Coordinate conventions
----------------------
Image space: x increases to the right, y increases downward; origin at
the top-left corner. Matches PIL/OpenCV.

World space: bearings in degrees clockwise from true north
(0=N, 90=E, 180=S, 270=W). GPS in decimal degrees, WGS84.

Limitations
-----------
- Distance estimation assumes a known real-world target height and a
  bounding box that tightly bounds the visible animal silhouette.
  Partial occlusion or edge-cropping biases distance estimates.
- The forward projection uses a spherical-Earth approximation
  (~0.3% error vs. WGS84 ellipsoidal Vincenty for typical
  camera-trap distances) — well below the dominant error source
  (bbox height noise).
- Assumes a flat ground plane and an undistorted rectilinear lens.
  Wide-angle traps with significant barrel distortion violate this
  assumption near the image edges.
"""

import math
from typing import Optional, Tuple

# WGS84 equatorial radius (meters). Used for the spherical-Earth forward
# projection. Switch to geographiclib.Geodesic for sub-meter precision
# on long baselines.
EARTH_RADIUS_M = 6_378_137.0


def focal_length_px(image_width: int, hfov_deg: float) -> float:
    """Compute focal length in pixels from image width and HFOV.

    Parameters
    ----------
    image_width : int
        Image width in pixels.
    hfov_deg : float
        Horizontal field of view in degrees.

    Returns
    -------
    float
        Focal length in pixels.
    """
    hfov_rad = math.radians(hfov_deg)
    return (image_width / 2.0) / math.tan(hfov_rad / 2.0)


def estimate_distance_and_angle(
    image_width: int,
    image_height: int,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    hfov_deg: float,
    real_height_m: float,
) -> Tuple[float, float]:
    """Estimate distance and horizontal angle from a bounding box.

    Distance is derived from the bounding box height assuming a known
    real-world target height. Angle is the horizontal offset of the box
    centroid from the camera's optical axis.

    Parameters
    ----------
    image_width, image_height : int
        Source image dimensions in pixels.
    x_min, y_min, x_max, y_max : float
        Bounding box in image pixels.
    hfov_deg : float
        Camera horizontal field of view in degrees.
    real_height_m : float
        Approximate real-world height of target species in meters.

    Returns
    -------
    distance_m : float
        Distance to the animal in meters; ``inf`` for invalid bboxes.
    angle_deg : float
        Horizontal offset angle in degrees. Positive = right of center.
    """
    bbox_h = y_max - y_min
    if bbox_h <= 0:
        return float("inf"), 0.0

    fx = focal_length_px(image_width, hfov_deg)
    distance_m = (real_height_m * fx) / bbox_h

    cx = (x_min + x_max) / 2.0
    dx = cx - (image_width / 2.0)
    angle_deg = math.degrees(math.atan(dx / fx))

    return distance_m, angle_deg


def project_gps(
    camera_lat: float,
    camera_lon: float,
    distance_m: float,
    absolute_bearing_deg: float,
) -> Tuple[float, float]:
    """Forward-project a target GPS coordinate from a camera position.

    Walks ``distance_m`` from ``(camera_lat, camera_lon)`` along the
    given absolute bearing using the spherical-Earth great-circle
    formula.

    Parameters
    ----------
    camera_lat, camera_lon : float
        Camera position, decimal degrees, WGS84.
    distance_m : float
        Distance to target, meters.
    absolute_bearing_deg : float
        Bearing in degrees clockwise from true north.

    Returns
    -------
    target_lat, target_lon : float
        Projected target position, decimal degrees, WGS84.
    """
    lat1 = math.radians(camera_lat)
    lon1 = math.radians(camera_lon)
    bearing = math.radians(absolute_bearing_deg)

    angular_dist = distance_m / EARTH_RADIUS_M

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_dist)
        + math.cos(lat1) * math.sin(angular_dist) * math.cos(bearing)
    )
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular_dist) * math.cos(lat1),
        math.cos(angular_dist) - math.sin(lat1) * math.sin(lat2),
    )

    return math.degrees(lat2), math.degrees(lon2)


def parse_camera_bearing(val) -> Optional[float]:
    """Parse a bearing input that may be a number or 'NA'.

    Returns
    -------
    Optional[float]
        Bearing normalized to [0, 360); ``None`` if input is missing or 'NA'.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.upper() == "NA":
        return None
    try:
        return float(s) % 360.0
    except (ValueError, TypeError):
        return None
