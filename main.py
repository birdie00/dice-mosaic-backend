from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance
import io
import os
from uuid import uuid4
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 🌐 Your deployed backend URL
RENDER_BASE_URL = "https://dice-mosaic-backend.onrender.com"

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace this with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Brightness & contrast adjustment
def apply_brightness_contrast(image: Image.Image, brightness=1.0, contrast=1.0):
    image = image.convert("RGB")
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image

# Dice map PDF generator
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

# Analyze endpoint
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    grid_width: int = Form(...),
    grid_height: int = Form(...),
    style_choice: int = Form(1)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Define brightness and contrast per style
    style_map = {
        1: (1.0, 1.0),  # Normal
        2: (1.1, 1.0),  # Brighter
        3: (0.9, 1.0),  # Darker
        4: (1.1, 1.1),  # Bright + Contrast
        5: (0.9, 1.1),  # Dark + Contrast
        6: (1.0, 1.1),  # Contrast Only
    }

    brightness, contrast = style_map.get(style_choice, (1.0, 1.0))
    print(f"📦 style_choice received: {style_choice}")
    print(f"🎛️ Applying brightness={brightness}, contrast={contrast}")

    # Apply adjustments
    processed_image = apply_brightness_contrast(image, brightness, contrast)

    # Save debug image
    debug_filename = f"debug_style_{style_choice}.jpg"
    debug_path = f"static/{debug_filename}"
    processed_image.save(debug_path)
    print(f"🖼️ Saved debug image: /static/{debug_filename}")

    # Flush and re-open image to ensure edits apply
    buffer = io.BytesIO()
    processed_image.save(buffer, format="PNG")
    buffer.seek(0)
    final_image = Image.open(buffer).convert("L")

    # Resize and convert pixels to dice values
    final_image = final_image.resize((grid_width, grid_height))
    pixels = list(final_image.getdata())
    values = [min(6, max(0, pixel // 40)) for pixel in pixels]
    grid = [values[i:i + grid_width] for i in range(0, len(values), grid_width)]

    print("🧪 Sample grayscale pixels:", pixels[:20])
    print("🎲 Sample dice values:", values[:20])

    # Generate dice map PDF
    pdf_filename = f"dice_map_{uuid4().hex}.pdf"
    pdf_path = f"static/{pdf_filename}"
    generate_dice_map_pdf(grid, pdf_path)

    return {
        "grid": grid,
        "dice_map_url": f"{RENDER_BASE_URL}/static/{pdf_filename}"
    }
