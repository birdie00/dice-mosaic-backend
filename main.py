
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
from reportlab.lib.colors import black, white
import numpy as np
import os
import cv2

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


def apply_enhancements(pil_img, brightness, contrast, sharpness, gamma=1.0, clahe=False):
    if clahe:
        cv_img = np.array(pil_img)
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cv_img = clahe_op.apply(cv_img)
        pil_img = Image.fromarray(cv_img)

    enhancer = ImageEnhance.Brightness(pil_img)
    pil_img = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(contrast)

    enhancer = ImageEnhance.Sharpness(pil_img)
    pil_img = enhancer.enhance(sharpness)

    if gamma != 1.0:
        arr = np.array(pil_img).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)

    return pil_img


@app.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    grid_width: int = Form(...),
    grid_height: int = Form(...),
):
    image = Image.open(file.file).convert("L").resize((grid_width, grid_height))

    style_settings = {
        1: {"brightness": 1.0, "contrast": 1.5, "sharpness": 2.0, "clahe": True, "gamma": 0.8},
        2: {"brightness": 1.1, "contrast": 1.2, "sharpness": 1.3, "clahe": True, "gamma": 0.9},
        3: {"brightness": 1.4, "contrast": 1.6, "sharpness": 1.5, "clahe": True, "gamma": 0.85},
        4: {"brightness": 0.6, "contrast": 1.8, "sharpness": 1.4, "clahe": True, "gamma": 1.0},
        5: {"brightness": 1.0, "contrast": 1.2, "sharpness": 1.3, "clahe": False, "gamma": 1.0},
        6: {"brightness": 0.8, "contrast": 1.3, "sharpness": 1.7, "clahe": True, "gamma": 0.9},
    }

    styles = []
    for style_id, settings in style_settings.items():
        processed = apply_enhancements(image.copy(), **settings)
        arr = np.array(processed)
        grid = [[int(val / 256 * 7) for val in row] for row in arr]
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
    font_size = 4.5
    label_font_size = 4

    # Dice value fill colors (can be modified for better visibility)
    fill_colors = {
        0: (30, 30, 30),
        1: (80, 80, 80),
        2: (140, 140, 140),
        3: (180, 180, 180),
        4: (200, 200, 200),
        5: (230, 230, 230),
        6: (255, 255, 255),
    }

    dice_count = {i: 0 for i in range(0, 7)}

    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    total_width = (cols + 1) * cell_size
    total_height = (rows + 1) * cell_size

    start_x = (page_width - total_width) / 2
    start_y = page_height - ((page_height - total_height) / 2)

    c = canvas.Canvas(filepath, pagesize=landscape(letter))

    # Draw title & instructions
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_width / 2, page_height - 30, "🎲 Pipcasso Dice Map")
    c.setFont("Helvetica", 8)
    c.drawCentredString(page_width / 2, page_height - 42, "Use this map to place dice according to the values below.")

    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            dice_count[val] += 1
            px = start_x + (x + 1) * cell_size
            py = start_y - (y + 1) * cell_size

            r, g, b = fill_colors.get(val, (255, 255, 255))
            c.setFillColorRGB(r / 255, g / 255, b / 255)
            c.rect(px, py, cell_size, cell_size, fill=1, stroke=0)

            # Draw white border
            c.setLineWidth(0.1)
            c.setStrokeColor(white)
            c.rect(px, py, cell_size, cell_size, fill=0, stroke=1)

            # Draw number
            c.setFont("Helvetica", font_size)
            c.setFillColor(white if val in [0, 1, 2] else black)
            c.drawCentredString(px + cell_size / 2, py + 1, str(val))

    # Row labels
    for y in range(rows):
        label = f"R{y + 1}"
        py = start_y - (y + 1) * cell_size
        c.setFont("Helvetica", label_font_size)
        c.setFillColor(black)
        c.setStrokeColor(black)
        c.rect(start_x, py, cell_size, cell_size, fill=0, stroke=1)
        c.drawCentredString(start_x + cell_size / 2, py + 1, label)

    # Column labels
    for x in range(cols):
        label = f"C{x + 1}"
        px = start_x + (x + 1) * cell_size
        py = start_y
        c.setFont("Helvetica", label_font_size)
        c.setFillColor(black)
        c.setStrokeColor(black)
        c.rect(px, py, cell_size, cell_size, fill=0, stroke=1)
        c.drawCentredString(px + cell_size / 2, py + 1, label)

    # Summary box (on second page)
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, page_height - margin / 2, "Dice Count Summary")
    for i, (val, count) in enumerate(sorted(dice_count.items())):
        c.drawString(margin, page_height - margin - (i + 1) * 14, f"Dice {val}: {count}")

    c.save()
    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})
