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
    number_font_size = 3.5  # reduced by 30%
    label_font_size = 3.0   # reduced by 50%

    color_map = {
        0: ((0, 0, 0), white),
        1: ((255, 0, 0), white),
        2: ((0, 0, 255), white),
        3: ((0, 128, 0), white),
        4: ((255, 165, 0), black),
        5: ((255, 255, 0), black),
        6: ((255, 255, 255), black),
    }

    dice_count = {i: 0 for i in range(0, 7)}

    rows = len(grid)
    cols = len(grid[0])
    rows_per_page = rows // 2

    c = canvas.Canvas(filepath, pagesize=landscape(letter))

    for page_num in range(2):
        start_row = page_num * rows_per_page
        end_row = min(start_row + rows_per_page, rows)

        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, page_height - margin / 2, "Pipcasso Dice Map")
        c.setFont("Helvetica", 10)
        c.drawString(margin, page_height - margin - 12, "Project: _____________")
        c.drawString(margin, page_height - margin - 24, f"Grid Size: {cols} x {rows}")
        c.drawString(margin, page_height - margin - 36, "Instructions: Match each square with the correct dice number and color.")

        grid_top = page_height - margin - 60
        grid_left = margin + cell_size

        # Column labels
        for x in range(cols):
            label = f"C{x + 1}"
            px = grid_left + x * cell_size
            py = grid_top + cell_size
            c.setFillColor(white)
            c.setStrokeColor(black)
            c.rect(px, py, cell_size, cell_size, fill=1)
            c.setFillColor(black)
            c.setFont("Helvetica", label_font_size)
            c.drawCentredString(px + cell_size / 2, py + 1.5, label)

        for y, row in enumerate(grid[start_row:end_row]):
            py = grid_top - y * cell_size
            row_idx = start_row + y
            # Row label
            label = f"R{row_idx + 1}"
            c.setFillColor(white)
            c.setStrokeColor(black)
            c.rect(grid_left - cell_size, py - cell_size, cell_size, cell_size, fill=1)
            c.setFillColor(black)
            c.setFont("Helvetica", label_font_size)
            c.drawCentredString(grid_left - cell_size / 2, py - cell_size + 1.5, label)

            for x, val in enumerate(row):
                px = grid_left + x * cell_size
                r, g, b = color_map.get(val, (255, 255, 255))[0]
                text_color = color_map.get(val, (255, 255, 255))[1]

                c.setFillColorRGB(r / 255, g / 255, b / 255)
                c.setStrokeColor(white)
                c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)

                c.setFillColor(text_color)
                c.setFont("Helvetica", number_font_size)
                c.drawCentredString(px + cell_size / 2, py - cell_size + 1.5, str(val))

        c.showPage()

    c.save()
    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})