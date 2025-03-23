from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance
import io
import os
from uuid import uuid4
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 🔗 Your deployed backend URL (no trailing slash)
RENDER_BASE_URL = "https://dice-mosaic-backend.onrender.com"

# Create static folder if not exist
os.makedirs("static", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Allow frontend to access this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 Apply brightness and contrast to image
def apply_brightness_contrast(image: Image.Image, brightness=1.0, contrast=1.0):
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image

# 🎲 Convert image to grid of dice values (0–6)
def generate_dice_grid(image: Image.Image, width: int, height: int):
    image = image.resize((width, height)).convert("L")  # convert to grayscale
    pixels = list(image.getdata())
    values = [min(6, max(0, pixel // 40)) for pixel in pixels]  # map brightness to 0–6
    return [values[i:i + width] for i in range(0, len(values), width)]

# 🧾 Generate PDF layout from dice grid
def generate_dice_map_pdf(grid, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    cell_size = 12
    margin = 40
    y_offset = height - margin

    c.setFont("Courier", 8)
    c.drawString(margin, y_offset, "Dice Map")
    y_offset -= 20

    for row in grid:
        x = margin
        for val in row:
            c.drawString(x, y_offset, str(val))
            x += cell_size
        y_offset -= cell_size

        if y_offset < margin:
            c.showPage()
            y_offset = height - margin

    c.save()

# 🚀 Main endpoint
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    grid_width: int = Form(...),
    grid_height: int = Form(...),
    style_choice: int = Form(1)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Define brightness & contrast for each style
    style_map = {
        1: (1.0, 1.0),  # No edit
        2: (1.1, 1.0),  # +Brightness
        3: (0.9, 1.0),  # -Brightness
        4: (1.1, 1.1),  # +Bright +Contrast
        5: (0.9, 1.1),  # -Bright +Contrast
        6: (1.0, 1.1),  # +Contrast
    }

    brightness, contrast = style_map.get(style_choice, (1.0, 1.0))
    print(f"▶️ Style {style_choice} → Brightness: {brightness}, Contrast: {contrast}")

    # Apply filters BEFORE generating grid
    processed_image = apply_brightness_contrast(image, brightness, contrast)

    # Generate dice values grid
    grid = generate_dice_grid(processed_image, grid_width, grid_height)

    # Generate and save dice map PDF
    filename = f"dice_map_{uuid4().hex}.pdf"
    pdf_path = f"static/{filename}"
    generate_dice_map_pdf(grid, pdf_path)

    # Return public URL and dice grid
    return {
        "grid": grid,
        "dice_map_url": f"{RENDER_BASE_URL}/static/{filename}"
    }
