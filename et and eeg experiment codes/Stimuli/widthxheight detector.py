import os
import csv
from PIL import Image

# Get the folder in which this script is stored
folder = os.path.dirname(os.path.abspath(__file__))
output_csv = os.path.join(folder, "image_sizes.csv")

rows = []

# Loop through all image files in the folder and collect their width and height
for filename in sorted(os.listdir(folder)):
    if filename.lower().endswith(".png") or filename.lower().endswith(".jpg"):
        filepath = os.path.join(folder, filename)

        try:
            with Image.open(filepath) as img:
                width, height = img.size
            rows.append([filename, width, height])
        except Exception as e:
            rows.append([filename, "ERROR", str(e)])

# Save the collected image sizes to a csv file
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file", "width", "height"])
    writer.writerows(rows)

print("Klaar:", output_csv)