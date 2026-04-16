# AOI Editor

A small local web app for drawing **Areas of Interest (AOIs)** on images and saving them to a CSV in real time.

## What it does

- Reads images from the local `images/` folder
- Shows the image list on the left
- Loads all saved AOIs whenever you open an image
- Lets you draw new rectangles by dragging
- Lets you drag existing rectangles to reposition them
- Saves to `aois.csv` with these columns:

```csv
image_name,name,x,y,width,height
```

Coordinates are stored in **original image pixels** from the **top-left corner**.

## Folder structure

```text
aoi_webapp/
├── app.py
├── aois.csv            # created automatically
├── images/             # put your images here
├── requirements.txt
├── static/
│   ├── app.js
│   └── styles.css
└── templates/
    └── index.html
```

## Run it

### 1) Create a virtual environment (recommended)

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Add your images

Copy image files into the `images/` folder.

### 4) Start the app

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## How to use

1. Click an image on the left.
2. Type the next AOI name in the top input.
3. Drag on empty space over the image to create a rectangle.
4. Drag an existing rectangle to move it.
5. Watch `aois.csv` update automatically.

If the AOI name box is empty when you finish drawing, the browser will prompt you for a name.

## Notes

- Supported image types: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.webp`, `.tif`, `.tiff`
- The CSV is rewritten whenever AOIs for the current image change, so it always stays in sync.
