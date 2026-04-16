from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock
from typing import Any

from flask import Flask, jsonify, render_template, request, send_from_directory


# Define the main folders/files used by the web app
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
CSV_PATH = BASE_DIR / "aois.csv"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
CSV_HEADERS = ["image_name", "name", "x", "y", "width", "height"]
CSV_LOCK = Lock()

app = Flask(__name__)


# Make sure the images folder and AOI csv file exist before using them
def ensure_setup() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            writer.writeheader()


# Get all valid image files from the images folder
def list_images() -> list[str]:
    ensure_setup()
    return sorted(
        [path.name for path in IMAGES_DIR.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS],
        key=str.lower,
    )


# Convert values safely to integers and keep them within a minimum value
def clean_int(value: Any, default: int = 0, minimum: int = 0) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = default
    return max(minimum, numeric)


# Standardize one AOI row so all fields have the right format
def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "image_name": str(row.get("image_name", "")).strip(),
        "name": str(row.get("name", "")).strip(),
        "x": clean_int(row.get("x"), minimum=0),
        "y": clean_int(row.get("y"), minimum=0),
        "width": clean_int(row.get("width"), default=1, minimum=1),
        "height": clean_int(row.get("height"), default=1, minimum=1),
    }


# Read all saved AOIs from the csv file
def read_aois() -> list[dict[str, Any]]:
    ensure_setup()
    with CSV_LOCK:
        with CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows: list[dict[str, Any]] = []
            for row in reader:
                if not row:
                    continue
                normalized = normalize_row(row)
                if normalized["image_name"]:
                    rows.append(normalized)
            return rows


# Save AOIs to the csv file in a clean and sorted way
def write_aois(rows: list[dict[str, Any]]) -> None:
    ensure_setup()
    ordered = sorted(
        [normalize_row(row) for row in rows if str(row.get("image_name", "")).strip()],
        key=lambda item: (item["image_name"].lower(), item["name"].lower(), item["y"], item["x"]),
    )
    with CSV_LOCK:
        with CSV_PATH.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            writer.writeheader()
            writer.writerows(ordered)


# Load the main webpage
@app.get("/")
def index() -> str:
    ensure_setup()
    return render_template("index.html")


# Serve image files from the images folder
@app.get("/images/<path:filename>")
def serve_image(filename: str):
    return send_from_directory(IMAGES_DIR, filename)


# Return the list of available images as JSON
@app.get("/api/images")
def get_images():
    return jsonify({"images": list_images()})


# Return all AOIs, or only the AOIs for one selected image
@app.get("/api/aois")
def get_aois():
    image_name = request.args.get("image_name", "").strip()
    rows = read_aois()
    if image_name:
        rows = [row for row in rows if row["image_name"] == image_name]
    return jsonify({"aois": rows})


# Save the AOIs for one image and replace older AOIs of that same image
@app.post("/api/aois/save_image")
def save_image_aois():
    payload = request.get_json(silent=True) or {}
    image_name = str(payload.get("image_name", "")).strip()
    incoming = payload.get("aois", [])

    if not image_name:
        return jsonify({"error": "image_name is required."}), 400
    if not isinstance(incoming, list):
        return jsonify({"error": "aois must be a list."}), 400

    existing_rows = read_aois()
    other_rows = [row for row in existing_rows if row["image_name"] != image_name]

    new_rows: list[dict[str, Any]] = []
    for row in incoming:
        if not isinstance(row, dict):
            continue
        normalized = normalize_row({**row, "image_name": image_name})
        if normalized["name"]:
            new_rows.append(normalized)

    write_aois(other_rows + new_rows)
    return jsonify({"ok": True, "saved": len(new_rows), "csv_path": str(CSV_PATH)})


# Start the local Flask web app
if __name__ == "__main__":
    ensure_setup()
    app.run(host="127.0.0.1", port=5001, debug=True)
