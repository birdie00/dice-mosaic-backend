from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
from uuid import uuid4
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 🔗 Render Base URL (change this to your actual Render URL)
RENDER_BASE_URL = "https://dice-mosaic-backend.onrender.com"

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# FastAPI app setup
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Generate grayscale dice grid
def generate_dice_grid(image: Image.Image, width: int, height: int):
    image = image.resize((width, height)).convert("L")  # Convert to grayscale
    pixels = list(image.getdata())
    values = [min(6, max(0, pixel // 40)) for pixel in pixels]  # Map brightness to 0–6
    return [values[i:i + width] for i in range(0, len(values), width)]

# 📄 Generate dice map PDF
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

# 🔍 Analyze endpoint
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    grid_width: int = Form(...),
    grid_height: int = Form(...)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Resize image based on grid width x height
    image = image.resize((grid_width, grid_height)).convert("L")

    # Generate the grid
    pixels = list(image.getdata())
    values = [min(6, max(0, pixel // 40)) for pixel in pixels]
    grid = [values[i:i + grid_width] for i in range(0, len(values), grid_width)]

    # Generate the PDF
    from uuid import uuid4
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    import os

    os.makedirs("static", exist_ok=True)
    filename = f"dice_map_{uuid4().hex}.pdf"
    pdf_path = f"static/{filename}"

    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    margin = 40
    y_offset = height - margin
    cell_size = 12

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

    dice_map_url = f"https://dice-mosaic-backend.onrender.com/static/{filename}"
    return {
        "grid": grid,
        "dice_map_url": dice_map_url
    }

