
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from uuid import uuid4
from PIL import Image, ImageEnhance
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.colors import black
import os
import numpy as np
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")


class GridRequest(BaseModel):
    grid_data: List[List[int]]


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    grid_width: int = Form(...),
    grid_height: int = Form(...),
):
    image = Image.open(file.file).convert("L").resize((grid_width, grid_height))

    def simulate_dice_map(img: Image.Image, brightness: float, contrast: float) -> List[List[int]]:
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        arr = np.array(img)
        return [[int(val / 256 * 7) for val in row] for row in arr]


    style_map = {
        1: (1.0, 1.0),
        2: (1.2, 1.0),
        3: (0.8, 1.0),
        4: (1.2, 1.2),
        5: (0.8, 1.2),
        6: (1.0, 1.2),
    }

    styles = []
    for style_id, (brightness, contrast) in style_map.items():
        grid = simulate_dice_map(image, brightness, contrast)
        styles.append({"style_id": style_id, "grid": grid})

    return JSONResponse(content={"styles": styles})


@app.post("/generate-pdf")
async def generate_dice_map_pdf(grid_data: GridRequest):
    grid = grid_data.grid_data
    filename = f"dice_map_{uuid4().hex}.pdf"
    filepath = os.path.join("static", filename)

    page_width, page_height = landscape(letter)
    cell_size = 6
    margin = 40
    font_size = 5

    colors = {
        0: (240, 240, 240),  # Light gray for dice_0
        1: (255, 255, 255),
        2: (200, 200, 200),
        3: (150, 150, 150),
        4: (100, 100, 100),
        5: (50, 50, 50),
        6: (0, 0, 0),
    }

    dice_count = {i: 0 for i in range(0, 7)}

    cols_per_page = int((page_width - 2 * margin) // cell_size)
    rows_per_page = int((page_height - 2 * margin) // cell_size)


    num_pages = int(np.ceil(len(grid) / rows_per_page))

    c = canvas.Canvas(filepath, pagesize=landscape(letter))

    dice_count = {i: 0 for i in range(1, 7)}

    for page_num in range(num_pages):
        start_row = page_num * rows_per_page
        end_row = min(start_row + rows_per_page, len(grid))

        for y, row in enumerate(grid[start_row:end_row]):
            for x, val in enumerate(row):
                if val not in colors:
                    continue
                px = margin + x * cell_size
                py = page_height - margin - y * cell_size
                r, g, b = colors[val]
                c.setFillColorRGB(r / 255, g / 255, b / 255)
                c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=0)
                c.setFillColor(black)
                c.setFont("Helvetica", font_size)
                c.drawCentredString(px + cell_size / 2, py - cell_size + 0.5, str(val))
                dice_count[val] += 1

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, margin / 2, f"Page {page_num + 1} of {num_pages}")
        c.showPage()

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, page_height - margin / 2, "Dice Count Summary")
    for i, (val, count) in enumerate(sorted(dice_count.items())):
        c.drawString(margin, page_height - margin - (i + 1) * 14, f"Dice {val}: {count}")

    c.save()

    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})
