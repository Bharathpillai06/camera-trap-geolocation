#!/usr/bin/env python3
"""YOLO + BioCLIP 2 (open_clip) + distance/angle + GPS forward projection.

Ultralytics YOLO single-pass detection, plus BioCLIP 2 zero-shot
classification against a user-supplied label list via ``open_clip``.

Dependencies::

    pip install ultralytics open_clip_torch torch pillow

Example::

    python scripts/detect_geolocate.py \\
        --weights yolov8n.pt \\
        --images ./photos \\
        --hfov-deg 52 \\
        --target-height-m 0.75 \\
        --camera-lat 39.83 \\
        --camera-lon -81.74 \\
        --camera-bearing-deg 180 \\
        --labels "african wild dog,dog,animal" \\
        --csv-out ./out/detections.csv
"""

import argparse
import csv
import datetime
import os

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

import torch
import open_clip

from camera_trap_geolocation import (
    estimate_distance_and_angle,
    project_gps,
    parse_camera_bearing,
    parse_iso_timestamp_from_filename,
    gather_images_recursive,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="YOLO detections + BioCLIP 2 (open_clip) + distance/angle + GPS"
    )

    # Paths / model
    p.add_argument("--weights", required=True, help="Path to YOLO weights (e.g., yolov8n.pt)")
    p.add_argument("--images", required=True, help="Directory containing images (recursive)")
    p.add_argument("--device", default="cpu", help='Device string, e.g. "cuda:0" or "cpu"')

    # YOLO inference params
    p.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")

    # Control
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on number of images (0 = no cap)",
    )
    p.add_argument(
        "--max-det-print",
        type=int,
        default=-1,
        help="Max number of detections per image to print (0 = none, -1 = all)",
    )

    # Geometry
    p.add_argument("--hfov-deg", type=float, required=True, help="Camera HFOV in degrees")
    p.add_argument(
        "--target-height-m",
        type=float,
        required=True,
        help="Approximate real-world height of target animal (meters)",
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

    # CSV output
    p.add_argument("--csv-out", type=str, required=True, help="Path to output CSV file")

    # Label set for BioCLIP 2 zero-shot
    p.add_argument(
        "--labels",
        type=str,
        default="white-tailed deer,deer,bird",
        help="Comma-separated list of candidate labels for BioCLIP 2 zero-shot",
    )

    return p.parse_args()


def load_bioclip_model(device: str, label_string: str):
    print("Loading BioCLIP 2 model from Hugging Face via open_clip...")
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2")

    model = model.to(device)
    model.eval()

    labels = [s.strip() for s in label_string.split(",") if s.strip()]
    if not labels:
        raise ValueError("No labels provided for BioCLIP 2.")

    text_prompts = [f"a photo of a {lbl}" for lbl in labels]
    with torch.no_grad():
        text_tokens = tokenizer(text_prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    print(f"Loaded BioCLIP 2 with {len(labels)} labels: {labels}")
    return model, preprocess_val, labels, text_features


def classify_with_bioclip(model, preprocess, text_features, labels, crop, device):
    with torch.no_grad():
        img_t = preprocess(crop).unsqueeze(0).to(device)
        image_features = model.encode_image(img_t)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.T).squeeze(0)
        best_idx = int(torch.argmax(logits).item())
        best_score = float(logits[best_idx].item())
        best_label = labels[best_idx]
    return best_label, best_score


def main():
    args = parse_args()

    assert os.path.isfile(args.weights), f"WEIGHTS not found: {args.weights}"
    assert os.path.isdir(args.images), f"IMAGES directory not found: {args.images}"

    camera_bearing = parse_camera_bearing(args.camera_bearing_deg)

    paths = gather_images_recursive(args.images)
    if not paths:
        raise SystemExit(f"No images found under {args.images} (recursive).")
    if args.max_images > 0:
        paths = paths[: args.max_images]

    print(f"Loading YOLO model from: {args.weights}")
    yolo_model = YOLO(args.weights)

    device_torch = args.device if args.device != "cpu" else "cpu"
    model_bio, preprocess_bio, bioclip_labels, text_features = load_bioclip_model(
        device_torch, args.labels
    )

    csv_rows = []

    total = len(paths)
    print(f"\nStarting scan on {total} image(s)")
    print(f"Distance model: target height = {args.target_height_m} m, HFOV = {args.hfov_deg} deg")
    if camera_bearing is None:
        print(
            f"Camera at lat={args.camera_lat}, lon={args.camera_lon}, "
            "bearing=NA (fallback animal lat/lon = camera)"
        )
    else:
        print(
            f"Camera at lat={args.camera_lat}, lon={args.camera_lon}, "
            f"bearing={camera_bearing} deg\n"
        )

    csv_dir = os.path.dirname(args.csv_out)
    annot_dir = os.path.join(csv_dir if csv_dir else ".", "annotated")
    os.makedirs(annot_dir, exist_ok=True)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for i, p in enumerate(paths, 1):
        fname = os.path.basename(p)
        print(f"[{i}/{total}] {fname}")

        try:
            img = Image.open(p).convert("RGB")
            img_width, img_height = img.size

            try:
                ts = os.path.getmtime(p)
                file_timestamp = datetime.datetime.fromtimestamp(ts).isoformat()
            except Exception:
                file_timestamp = None

        except Exception as e:
            print(f"  [SKIP] Could not open image: {e}")
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

            x_min_cl = max(0, int(x_min))
            y_min_cl = max(0, int(y_min))
            x_max_cl = min(img_width, int(x_max))
            y_max_cl = min(img_height, int(y_max))
            if x_max_cl <= x_min_cl or y_max_cl <= y_min_cl:
                continue

            crop = img.crop((x_min_cl, y_min_cl, x_max_cl, y_max_cl))
            bioclip_label, bioclip_score = classify_with_bioclip(
                model_bio, preprocess_bio, text_features, bioclip_labels, crop, device_torch
            )

            kept_det_idx += 1

            d_m, a_deg = estimate_distance_and_angle(
                img_width, img_height, x_min, y_min, x_max, y_max,
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

            if args.max_det_print != 0:
                if args.max_det_print < 0 or kept_det_idx <= args.max_det_print:
                    abs_str = (
                        f"{absolute_bearing:.1f} deg"
                        if absolute_bearing is not None
                        else "NA"
                    )
                    print(
                        f"   #{kept_det_idx} {bioclip_label} | yolo={yolo_conf:.2f} | "
                        f"bio={bioclip_score:.3f} | d={d_m:.1f} m | "
                        f"rel={a_deg:+.1f} deg | abs={abs_str}"
                    )

            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)], outline="red", width=2
            )
            label_text = f"{bioclip_label} ({bioclip_score:.2f})"
            text_x, text_y = x_min, max(0, y_min - 14)
            if font is not None:
                draw.text((text_x, text_y), label_text, fill="red", font=font)
            else:
                draw.text((text_x, text_y), label_text, fill="red")

            csv_rows.append(
                {
                    "image": fname,
                    "image_path": p,
                    "timestamp_file": file_timestamp,
                    "timestamp_from_name": timestamp_from_name,
                    "detection_index": kept_det_idx,
                    "bioclip_label": bioclip_label,
                    "bioclip_score": bioclip_score,
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
            )

        base, _ = os.path.splitext(fname)
        out_img_path = os.path.join(annot_dir, f"{base}_annotated.jpg")
        img.save(out_img_path, format="JPEG")

    if csv_rows:
        fieldnames = [
            "image", "image_path", "timestamp_file", "timestamp_from_name",
            "detection_index", "bioclip_label", "bioclip_score", "yolo_conf",
            "xmin", "ymin", "xmax", "ymax",
            "distance_m", "angle_deg",
            "camera_lat", "camera_lon", "camera_bearing_deg",
            "animal_lat", "animal_lon", "animal_absolute_bearing_deg",
        ]

        out_dir = os.path.dirname(args.csv_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nDetections saved to CSV: {args.csv_out}")
        print(f"Annotated images in:    {annot_dir}")
    else:
        print("\nNo detections found - CSV not written.")


if __name__ == "__main__":
    main()
