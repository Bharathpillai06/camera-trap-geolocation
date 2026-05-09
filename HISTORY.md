# History

All user-facing changes to this project are documented in this file.
The format is loosely based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-05-08

### Added

- Initial release.
- `scripts/detect_geolocate_sahi_openclip.py`: full geolocation
  pipeline using YOLO+SAHI sliced inference and BioCLIP 2 via
  `open_clip` with a user-supplied zero-shot label list.
- `src/camera_trap_geolocation/`: shared library with
  `geometry.estimate_distance_and_angle`, `geometry.project_gps`,
  `timestamps.parse_iso_timestamp_from_filename`, and
  `io_utils.gather_images_recursive`.
- Unit tests for the geometry and timestamp parsing modules.
