from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageEnhance
from uuid import uuid4
import io
import os
import json
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

# Base URL for linking to static files
RENDER_BASE_URL = "https://dice-mosaic-backend.onrender.com"

# Prepare FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static", exist_ok=True)

# Map brightness & contrast styles
style_map = {
    1: (1.0, 1.0),
    2: (1.2, 1.0),
    3: (0.8, 1.0),
    4: (1.2, 1.2),
    5: (0.8, 1.2),
    6: (1.0, 1.2),
}

# Generate grid from image
def generate_dice_grid(image: Image.Image, grid_width: int, grid_height: int):
    image = image.resize((grid_width, grid_height)).convert("L")
    pixels = list(image.getdata())
    values = [min(6, max(1, pixel // 40)) for pixel in pixels]
    return [values[i:i + grid_width] for i in range(0, len(values), grid_width)]

# Generate color-coded PDF
def generate_dice_map_pdf(grid, output_path):
    c = canvas.Canvas(output_path, pagesize=landscape(letter))
    width, height = landscape(letter)

    margin = 40
    cell_size = 16
    font_size = 8
    x_offset = margin
    y_offset = height - margin

    dice_colors = {
        1: colors.lightgrey,
        2: colors.lightblue,
        3: colors.lightgreen,
        4: colors.khaki,
        5: colors.salmon,
        6: colors.plum,
    }

    # Draw title and instructions
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_offset, "Dice Build Map")
    c.setFont("Helvetica", 10)
    y_offset -= 20
    c.drawString(margin, y_offset, "Use the numbers and colors to place the dice according to the layout.")
    y_offset -= 30

    start_y = y_offset

    # Draw column headers
    for col in range(len(grid[0])):
        c.setFont("Helvetica", font_size)
        c.drawString(x_offset + cell_size * (col + 1) + 3, start_y + 3, str(col + 1))

    # Draw rows
    dice_counts = {i: 0 for i in range(1, 7)}
    for row_idx, row in enumerate(grid):
        y = start_y - cell_size * (row_idx + 1)
        c.setFont("Helvetica", font_size)
        c.drawString(x_offset, y + 4, str(row_idx + 1))  # Row number

        for col_idx, val in enumerate(row):
            dice_counts[val] += 1
            x = x_offset + cell_size * (col_idx + 1)
            c.setFillColor(dice_colors[val])
            c.rect(x, y, cell_size, cell_size, fill=1, stroke=1)
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", font_size)
            c.drawCentredString(x + cell_size / 2, y + 4, str(val))

    # Draw dice count legend
    legend_y = margin + 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, legend_y, "Dice Count Summary:")

    legend_y -= 20
    for i in range(1, 7):
        c.setFillColor(dice_colors[i])
        c.rect(margin, legend_y, 12, 12, fill=1, stroke=1)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)
        c.drawString(margin + 20, legend_y + 2, f"Dice {i}: {dice_counts[i]} pieces")
        legend_y -= 16

    c.save()

# Analyze endpoint
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    grid_width: int = Form(...),
    grid_height: int = Form(...),
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    style_outputs = []
    for style_id, (brightness, contrast) in style_map.items():
        styled = image.copy()
        styled = ImageEnhance.Brightness(styled).enhance(brightness)
        styled = ImageEnhance.Contrast(styled).enhance(contrast)
        grid = generate_dice_grid(styled, grid_width, grid_height)

        style_outputs.append({
            "style_id": style_id,
            "grid": grid
        })

    return {"styles": style_outputs}

# PDF export endpoint
@app.post("/generate-pdf")
async def generate_pdf(grid_data: str = Form(...)):
    grid = json.loads(grid_data)
    filename = f"dice_map_{uuid4().hex}.pdf"
    pdf_path = f"static/{filename}"
    generate_dice_map_pdf(grid, pdf_path)
    return {"dice_map_url": f"{RENDER_BASE_URL}/static/{filename}"}
