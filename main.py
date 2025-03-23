from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
from uuid import uuid4
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 🔗 Replace with your actual Render backend URL (no trailing slash)
RENDER_BASE_URL = "https://dice-mosaic-backend.onrender.com"

# Setup static folder to host PDF files
os.makedirs("static", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS setup (optional, but recommended for frontend-backend communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✏️ Function to generate the dice grid — placeholder logic
def generate_dice_grid(image, grid_size):
    # You must implement this based on your mosaic logic
    # For example: resize image, convert to grayscale, threshold values to 0–6
    width, height = image.size
    image = image.resize((grid_size, grid_size)).convert("L")
    pixels = list(image.getdata())
    values = [min(6, max(0, pixel // 40)) for pixel in pixels]  # Map 0–255 to 0–6
    return [values[i:i + grid_size] for i in range(0, len(values), grid_size)]

# ✏️ Function to generate the dice map PDF
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

# ✅ Main analyze endpoint
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), grid_size: int = 40):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # 1. Generate the dice grid
    grid = generate_dice_grid(image, grid_size)

    # 2. Save PDF with unique filename
    filename = f"dice_map_{uuid4().hex}.pdf"
    pdf_path = f"static/{filename}"
    generate_dice_map_pdf(grid, pdf_path)

    # 3. Build public URL
    dice_map_url = f"{RENDER_BASE_URL}/static/{filename}"

    # 4. Return response
    return {
        "grid": grid,
        "dice_map_url": dice_map_url
    }
