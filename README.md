# camera-trap-geolocation

Estimate the GPS coordinates and bearing of animals detected in
camera-trap imagery, using only the camera's known position, the
camera's known compass bearing, and a pinhole geometric projection
from the bounding box.

The pipeline pairs **YOLO + SAHI** (sliced inference, for recall on
small or distant animals) with **BioCLIP 2** zero-shot classification
via `open_clip`, then projects each detection's bounding box to a
real-world GPS coordinate.

Each per-detection row of the output CSV includes the estimated animal
latitude and longitude alongside species labels, distances, and
bounding boxes.

## How the geolocation works

Given a detection with bounding box `(x_min, y_min, x_max, y_max)` in
an image of width `W` and height `H`, taken by a camera at known
`(camera_lat, camera_lon)` with a known horizontal field of view
`HFOV` and a known compass `bearing`, the pipeline:

1. Computes focal length in pixels: `fx = (W/2) / tan(HFOV/2)`.
2. Estimates distance from the bounding-box height assuming a known
   real-world target height `h_real` (e.g. 0.9 m for a deer):
   `d = h_real * fx / bbox_h`.
3. Computes the relative horizontal angle of the bbox center:
   `θ = arctan((cx − W/2) / fx)`.
4. Projects the animal's GPS coordinates by walking distance `d` from
   the camera at absolute bearing `(camera_bearing + θ) mod 360°`.

The projection uses the spherical-Earth forward formula on WGS84.
Accuracy is best for mid-range detections (10–150 m). At long range,
small bounding boxes amplify distance error.

## Installation

Requires Python 3.10+ and (for GPU inference) a CUDA-enabled PyTorch
install.

```bash
git clone https://github.com/Bharathpillai06/camera-trap-geolocation.git
cd camera-trap-geolocation
pip install -r requirements.txt
pip install -e .
```

You will also need a YOLO weights file (e.g. `yolov8x.pt`); download
from the
[Ultralytics releases](https://github.com/ultralytics/assets/releases).

## Usage

```bash
python scripts/detect_geolocate_sahi_openclip.py \
    --weights yolov8x.pt \
    --images ./photos/site_A \
    --hfov-deg 60 \
    --target-height-m 0.9 \
    --camera-lat 40.123 \
    --camera-lon -83.456 \
    --camera-bearing-deg 270 \
    --labels "white-tailed deer,deer,bird" \
    --csv-out ./out/site_A.csv
```

When the camera bearing is unknown, pass `--camera-bearing-deg NA`.
The pipeline then falls back to the camera position as a conservative
estimate for `animal_lat` / `animal_lon`.

## Input format

Image filenames following the pattern
`<prefix>_YYMMDDHHMMSS_<suffix>.jpg` have their timestamp parsed
automatically (e.g. `NSCF0001_240714183022_0088.JPG`). Other filenames
are still processed; their timestamp is taken from the file
modification time and recorded as `timestamp_file`.

GPS coordinates are passed on the command line per camera. For
multi-site deployments, run the script once per site with that site's
camera parameters.

## Output

The CSV contains one row per detection. The full schema is documented
in [`data/README.md`](data/README.md). Annotated copies of every input
image (with bounding boxes drawn) are saved to an `annotated/`
subdirectory next to the CSV.

## Sources and acknowledgments

This tool was developed in support of the
[SmartWilds](https://huggingface.co/collections/imageomics/smartwilds)
multimodal wildlife monitoring dataset and was inspired by the
geometric localization approach described in the SmartWilds analysis
pipeline.

The pipeline depends on:

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for object detection
- [SAHI](https://github.com/obss/sahi) for sliced inference on small targets
- [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) for biological image classification
- [open_clip](https://github.com/mlfoundations/open_clip) for the BioCLIP 2 inference backend

## Citation

If you use this tool in your research, please cite both this
repository and the BioCLIP 2 paper. See
[`CITATION.cff`](CITATION.cff) for the machine-readable citation;
GitHub surfaces a "Cite this repository" button on the repo's main
page.

## License

This project is licensed under the MIT License — see
[`LICENSE.md`](LICENSE.md).

## Acknowledgments

This work was supported by the Imageomics Institute, funded by the US
National Science Foundation's Harnessing the Data Revolution (HDR)
program under Award #2118240.
