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
    image = image.convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image

def generate_dice_grid(image, grid_width, grid_height):
    image = image.resize((grid_width, grid_height)).convert("L")
    pixels = list(image.getdata())
    values = [min(6, max(0, pixel // 40)) for pixel in pixels]
    return [values[i:i + grid_width] for i in range(0, len(values), grid_width)]

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
    grid_height: int = Form(...)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    style_map = {
    1: (1.0, 1.0),   # no change
    2: (1.2, 1.0),   # +20% brightness
    3: (0.8, 1.0),   # -20% brightness
    4: (1.2, 1.2),   # +20% brightness & +20% contrast
    5: (0.8, 1.2),   # -20% brightness & +20% contrast
    6: (1.0, 1.2),   # +20% contrast
}

    results = []

    for style_id, (brightness, contrast) in style_map.items():
        styled_img = apply_brightness_contrast(image, brightness, contrast)
        buffer = io.BytesIO()
        styled_img.save(buffer, format="PNG")
        buffer.seek(0)
        gray_img = Image.open(buffer).convert("L")

        grid = generate_dice_grid(gray_img, grid_width, grid_height)
        preview_filename = f"mosaic_preview_{style_id}_{uuid4().hex}.png"
        preview_path = f"static/{preview_filename}"

        # Save mosaic preview image
        mosaic_img = Image.new("L", (grid_width, grid_height))
        mosaic_img.putdata([val * 40 for row in grid for val in row])
        mosaic_img = mosaic_img.resize((grid_width * 6, grid_height * 6), resample=Image.NEAREST)
        mosaic_img.save(preview_path)

        results.append({
            "style_id": style_id,
            "grid": grid,
            "preview_url": f"{RENDER_BASE_URL}/static/{preview_filename}"
        })

    return {"styles": results}

@app.post("/generate-pdf")
async def generate_pdf(
    grid_data: str = Form(...),
):
    import json
    grid = json.loads(grid_data)

    filename = f"dice_map_{uuid4().hex}.pdf"
    pdf_path = f"static/{filename}"
    generate_dice_map_pdf(grid, pdf_path)

    return {"dice_map_url": f"{RENDER_BASE_URL}/static/{filename}"}
