from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance
import io
import os
from uuid import uuid4
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

RENDER_BASE_URL = "https://dice-mosaic-backend.onrender.com"

os.makedirs("static", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def apply_brightness_contrast(image: Image.Image, brightness=1.0, contrast=1.0):
    enhancer_b = ImageEnhance.Brightness(image)
    image = enhancer_b.enhance(brightness)

    enhancer_c = ImageEnhance.Contrast(image)
    image = enhancer_c.enhance(contrast)

    return image

def generate_dice_grid(image: Image.Image, width: int, height: int):
    image = image.resize((width, height)).convert("L")  # grayscale
    pixels = list(image.getdata())
    values = [min(6, max(0, pixel // 40)) for pixel in pixels]
    return [values[i:i + width] for i in range(0, len(values), width)]

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

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    grid_width: int = Form(...),
    grid_height: int = Form(...),
    style_choice: int = Form(1)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Brightness and contrast map based on style
    style_map = {
        1: (1.0, 1.0),
        2: (1.1, 1.0),
        3: (0.9, 1.0),
        4: (1.1, 1.1),
        5: (0.9, 1.1),
        6: (1.0, 1.1),
    }
    brightness, contrast = style_map.get(style_choice, (1.0, 1.0))

    # Apply style
    image = apply_brightness_contrast(image, brightness, contrast)

    # Generate dice grid
    grid = generate_dice_grid(image, grid_width, grid_height)

    # Generate PDF
    filename = f"dice_map_{uuid4().hex}.pdf"
    pdf_path = f"static/{filename}"
    generate_dice_map_pdf(grid, pdf_path)

    return {
        "grid": grid,
        "dice_map_url": f"{RENDER_BASE_URL}/static/{filename}"
    }
