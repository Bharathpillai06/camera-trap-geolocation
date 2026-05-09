# Data formats

## Input: image directories

The geolocation scripts walk a directory of camera-trap images recursively
and process every `.jpg`, `.jpeg`, or `.png` they find. Subfolder
structure is preserved in the `image_path` column of the output CSV.

Filenames matching the SmartWilds convention have their timestamp
parsed automatically:

```
NSCF0001_240714183022_0088.JPG
         └─YYMMDDHHMMSS─┘
```

The example above resolves to `2024-07-14T18:30:22`. Filenames that
don't match this pattern still process; their timestamp is taken from
the file modification time and recorded as `timestamp_file`.

## Input: camera parameters

These are passed on the command line, one set per camera:

| Argument | Type | Notes |
|---|---|---|
| `--hfov-deg` | float | Horizontal field of view of the camera lens in degrees. Typical trail cams: 42–60°. Check the model spec sheet. |
| `--target-height-m` | float | Approximate real-world height of the species you expect to detect, in meters. Examples: white-tailed deer ≈ 0.9 m at the shoulder; coyote ≈ 0.6 m; black bear ≈ 0.9 m. |
| `--camera-lat`, `--camera-lon` | float | Camera GPS in decimal degrees, WGS84. Record at deployment. |
| `--camera-bearing-deg` | float or `NA` | Compass direction the lens points, degrees clockwise from true north. Pass `NA` if unknown — the pipeline will fall back to placing detections at the camera's coordinates. |

For multi-camera deployments, run the script once per camera with that
camera's parameters. The output CSVs from multiple cameras can be
concatenated downstream.

## Output: detection CSV

One row per detection. Columns from the `pybioclip` pipeline:

| Column | Description |
|---|---|
| `image` | Filename (no path) |
| `image_path` | Absolute or relative path used at runtime |
| `timestamp_file` | File modification time, ISO 8601 |
| `timestamp_from_name` | Parsed from filename if SmartWilds-format, else empty |
| `detection_index` | 1-indexed detection number within this image |
| `bioclip_rank` | Taxonomic rank requested at runtime |
| `bioclip_label` | Top BioCLIP 2 label at the requested rank |
| `bioclip_score` | Top label cosine-similarity score |
| `common_name` | English common name from the Tree of Life entry |
| `kingdom`, `phylum`, `class`, `order`, `family`, `genus`, `species` | Full Linnaean taxonomy of the top hit |
| `yolo_conf` | YOLO detector confidence for this bounding box |
| `xmin`, `ymin`, `xmax`, `ymax` | Bounding box in image pixels |
| `distance_m` | Estimated distance from camera (meters) |
| `angle_deg` | Horizontal angle from camera optical axis (degrees, + = right) |
| `camera_lat`, `camera_lon` | Camera position (echoed for convenience) |
| `camera_bearing_deg` | Camera compass bearing or `NA` |
| `animal_lat`, `animal_lon` | **Estimated animal GPS** (decimal degrees, WGS84) |
| `animal_absolute_bearing_deg` | Camera bearing + relative angle (degrees) |

When `--top-k > 1`, runner-up predictions appear as additional columns
named `bioclip_label_2`, `bioclip_score_2`, etc.

The `sahi_openclip` pipeline output has the same shape minus the full
taxonomy columns (it only emits `bioclip_label` from the user-supplied
label list).

## Output: annotated images

Alongside the CSV, the script writes annotated copies of every input
image into a directory named `annotated/` next to the CSV. Each
detection is drawn as a red bounding box with its top label and score.
Use these for spot-checking the model output before running downstream
analysis.
