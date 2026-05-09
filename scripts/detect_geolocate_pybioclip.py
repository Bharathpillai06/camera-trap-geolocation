#!/usr/bin/env python3
"""YOLO + BioCLIP 2 TreeOfLifeClassifier + distance/angle + GPS forward projection.

Uses ``pybioclip`` for full Linnaean taxonomy classification (kingdom -> species).
Each detection is classified against BioCLIP 2's built-in Tree of Life
index — no label list needed.

Dependencies::

    pip install ultralytics pybioclip pillow

Example::

    python scripts/detect_geolocate_pybioclip.py \\
        --weights yolov8n.pt \\
        --images ./photos \\
        --hfov-deg 60 \\
        --target-height-m 0.9 \\
        --camera-lat 40.123 \\
        --camera-lon -83.456 \\
        --camera-bearing-deg 270 \\
        --csv-out ./results/detections.csv \\
        --rank species \\
        --top-k 3
"""

import argparse
import csv
import datetime
import os
import tempfile

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from bioclip import TreeOfLifeClassifier, Rank

from camera_trap_geolocation import (
    estimate_distance_and_angle,
    project_gps,
    parse_camera_bearing,
    parse_iso_timestamp_from_filename,
    gather_images_recursive,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO + BioCLIP 2 (pybioclip) + distance/angle + GPS"
    )

    # Paths / model
    p.add_argument("--weights", required=True, help="Path to YOLO weights (e.g. yolov8n.pt)")
    p.add_argument("--images", required=True, help="Directory containing images (recursive)")
    p.add_argument("--device", default="cpu", help='Device: "cpu", "cuda:0", or "mps"')

    # YOLO inference params
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")

    # Image cap / print control
    p.add_argument("--max-images", type=int, default=0, help="Cap on images to process (0 = no cap)")
    p.add_argument(
        "--max-det-print",
        type=int,
        default=-1,
        help="Max detections per image to print (0 = none, -1 = all)",
    )

    # BioCLIP 2 taxonomy settings
    p.add_argument(
        "--rank",
        type=str,
        default="species",
        choices=["kingdom", "phylum", "class", "order", "family", "genus", "species"],
        help="Taxonomic rank to classify at (default: species)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top BioCLIP predictions to record per detection (default: 1)",
    )

    # Geometry
    p.add_argument("--hfov-deg", type=float, required=True, help="Camera HFOV in degrees")
    p.add_argument(
        "--target-height-m",
        type=float,
        required=True,
        help="Approximate real-world height of the target species in metres",
    )

    # Camera GPS & direction
    p.add_argument("--camera-lat", type=float, required=True, help="Camera latitude (decimal degrees)")
    p.add_argument("--camera-lon", type=float, required=True, help="Camera longitude (decimal degrees)")
    p.add_argument(
        "--camera-bearing-deg",
        type=str,
        required=True,
        help='Camera bearing degrees clockwise from North (0-360), or "NA"',
    )

    # Output
    p.add_argument("--csv-out", type=str, required=True, help="Path to output CSV file")

    return p.parse_args()


def rank_from_str(rank_str: str) -> Rank:
    return {
        "kingdom": Rank.KINGDOM,
        "phylum": Rank.PHYLUM,
        "class": Rank.CLASS,
        "order": Rank.ORDER,
        "family": Rank.FAMILY,
        "genus": Rank.GENUS,
        "species": Rank.SPECIES,
    }[rank_str.lower()]


def format_prediction(pred: dict, rank_str: str):
    """Extract label at requested rank and its score from a pybioclip prediction."""
    label = pred.get(rank_str, pred.get("species", "unknown"))
    score = float(pred.get("score", 0.0))
    return label, score


def main():
    args = parse_args()

    assert os.path.isfile(args.weights), f"Weights not found: {args.weights}"
    assert os.path.isdir(args.images), f"Images directory not found: {args.images}"

    camera_bearing = parse_camera_bearing(args.camera_bearing_deg)
    rank_enum = rank_from_str(args.rank)

    paths = gather_images_recursive(args.images)
    if not paths:
        raise SystemExit(f"No images found under {args.images}")
    if args.max_images > 0:
        paths = paths[: args.max_images]

    print(f"Loading YOLO model: {args.weights}")
    yolo_model = YOLO(args.weights)

    print(f"Loading BioCLIP 2 TreeOfLifeClassifier (rank={args.rank}, top-k={args.top_k})...")
    classifier = TreeOfLifeClassifier(
        device=args.device,
        model_str="hf-hub:imageomics/bioclip-2",
    )
    print("BioCLIP 2 ready.")

    csv_dir = os.path.dirname(args.csv_out)
    annot_dir = os.path.join(csv_dir if csv_dir else ".", "annotated")
    os.makedirs(annot_dir, exist_ok=True)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    csv_rows = []

    print(f"\nStarting scan on {len(paths)} image(s)")
    print(f"Distance model: target height={args.target_height_m} m, HFOV={args.hfov_deg} deg")
    bearing_str = f"{camera_bearing} deg" if camera_bearing is not None else "NA"
    print(f"Camera: lat={args.camera_lat}, lon={args.camera_lon}, bearing={bearing_str}\n")

    for i, p in enumerate(paths, 1):
        fname = os.path.basename(p)
        print(f"[{i}/{len(paths)}] {fname}")

        try:
            img = Image.open(p).convert("RGB")
            img_w, img_h = img.size
            try:
                file_timestamp = datetime.datetime.fromtimestamp(
                    os.path.getmtime(p)
                ).isoformat()
            except Exception:
                file_timestamp = None
        except Exception as e:
            print(f"  [SKIP] Could not open: {e}")
            continue

        timestamp_from_name = parse_iso_timestamp_from_filename(fname)

        results = yolo_model.predict(
            source=p,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )

        draw = ImageDraw.Draw(img)
        kept_det_idx = 0

        for det in results[0].boxes:
            x_min, y_min, x_max, y_max = [float(v) for v in det.xyxy[0]]
            yolo_conf = float(det.conf[0])

            x0 = max(0, int(x_min))
            y0 = max(0, int(y_min))
            x1 = min(img_w, int(x_max))
            y1 = min(img_h, int(y_max))
            if x1 <= x0 or y1 <= y0:
                continue

            crop = img.crop((x0, y0, x1, y1))

            # pybioclip needs a file path, so save the crop to a temp file.
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tmp_path = tf.name
            try:
                crop.save(tmp_path, format="JPEG")
                predictions = classifier.predict(tmp_path, rank_enum, k=args.top_k)
            finally:
                os.unlink(tmp_path)

            top_pred = predictions[0] if predictions else {}
            top_label, top_score = format_prediction(top_pred, args.rank)
            top_kingdom = top_pred.get("kingdom", "")
            top_phylum = top_pred.get("phylum", "")
            top_class = top_pred.get("class", "")
            top_order = top_pred.get("order", "")
            top_family = top_pred.get("family", "")
            top_genus = top_pred.get("genus", "")
            top_species = top_pred.get("species", "")
            top_common_name = top_pred.get("common_name", "")

            kept_det_idx += 1

            d_m, a_deg = estimate_distance_and_angle(
                img_w, img_h, x_min, y_min, x_max, y_max,
                args.hfov_deg, args.target_height_m,
            )

            if camera_bearing is None:
                absolute_bearing = None
                animal_lat, animal_lon = args.camera_lat, args.camera_lon
            else:
                absolute_bearing = (camera_bearing + a_deg) % 360.0
                animal_lat, animal_lon = project_gps(
                    args.camera_lat, args.camera_lon, d_m, absolute_bearing
                )

            if args.max_det_print != 0 and (
                args.max_det_print < 0 or kept_det_idx <= args.max_det_print
            ):
                abs_str = (
                    f"{absolute_bearing:.1f} deg"
                    if absolute_bearing is not None
                    else "NA"
                )
                print(
                    f"  #{kept_det_idx} {top_label} ({top_common_name}) | "
                    f"yolo={yolo_conf:.2f} | bio={top_score:.3f} | "
                    f"d={d_m:.1f} m | rel={a_deg:+.1f} deg | abs={abs_str}"
                )

            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)], outline="red", width=2
            )
            label_text = f"{top_label} ({top_score:.2f})"
            draw.text(
                (x_min, max(0, y_min - 14)),
                label_text,
                fill="red",
                font=font,
            )

            row = {
                "image": fname,
                "image_path": p,
                "timestamp_file": file_timestamp,
                "timestamp_from_name": timestamp_from_name,
                "detection_index": kept_det_idx,
                "bioclip_rank": args.rank,
                "bioclip_label": top_label,
                "bioclip_score": top_score,
                "common_name": top_common_name,
                "kingdom": top_kingdom,
                "phylum": top_phylum,
                "class": top_class,
                "order": top_order,
                "family": top_family,
                "genus": top_genus,
                "species": top_species,
                "yolo_conf": yolo_conf,
                "xmin": x_min,
                "ymin": y_min,
                "xmax": x_max,
                "ymax": y_max,
                "distance_m": d_m,
                "angle_deg": a_deg,
                "camera_lat": args.camera_lat,
                "camera_lon": args.camera_lon,
                "camera_bearing_deg": (
                    camera_bearing if camera_bearing is not None else "NA"
                ),
                "animal_lat": animal_lat,
                "animal_lon": animal_lon,
                "animal_absolute_bearing_deg": (
                    absolute_bearing if absolute_bearing is not None else "NA"
                ),
            }

            for rank_idx, pred in enumerate(predictions[1:], start=2):
                lbl, sc = format_prediction(pred, args.rank)
                row[f"bioclip_label_{rank_idx}"] = lbl
                row[f"bioclip_score_{rank_idx}"] = sc

            csv_rows.append(row)

        base, _ = os.path.splitext(fname)
        img.save(os.path.join(annot_dir, f"{base}_annotated.jpg"), format="JPEG")

    if csv_rows:
        fieldnames = list(dict.fromkeys(k for row in csv_rows for k in row))

        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nDetections saved to: {args.csv_out}")
        print(f"Annotated images in: {annot_dir}")
    else:
        print("\nNo detections found - CSV not written.")


if __name__ == "__main__":
    main()
