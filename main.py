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
    font_size = 4
    label_font_size = 3

    colors = {
        0: (0, 0, 0),
        1: (255, 255, 255),
        2: (200, 200, 200),
        3: (150, 150, 150),
        4: (100, 100, 100),
        5: (50, 50, 50),
        6: (0, 0, 0),
    }

    text_colors = {
        0: white,
        1: white,
        2: white,
        3: black,
        4: black,
        5: black,
        6: white,
    }

    dice_count = {i: 0 for i in range(0, 7)}

    cols_per_page = int((page_width - 2 * margin) // cell_size)
    rows_per_page = int((page_height - 2 * margin) // cell_size)

    num_pages = int(np.ceil(len(grid) / rows_per_page))
    c = canvas.Canvas(filepath, pagesize=landscape(letter))

    for page_num in range(num_pages):
        start_row = page_num * rows_per_page
        end_row = min(start_row + rows_per_page, len(grid))

        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, page_height - margin / 2, "Pipcasso Dice Map")

        for y, row in enumerate(grid[start_row:end_row]):
            y_offset = y * cell_size
            for x, val in enumerate(row):
                x_offset = x * cell_size
                px = margin + x_offset
                py = page_height - margin - y_offset

                r, g, b = colors[val]
                c.setStrokeColorRGB(1, 1, 1)  # white border
                c.setLineWidth(0.2)
                c.setFillColorRGB(r / 255, g / 255, b / 255)
                c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)

                c.setFont("Helvetica", font_size)
                c.setFillColor(text_colors[val])
                c.drawCentredString(px + cell_size / 2, py - cell_size + 0.5, str(val))
                dice_count[val] += 1

        # Row labels (R1, R2, ...)
        c.setFont("Helvetica", label_font_size)
        for y in range(end_row - start_row):
            label = f"R{start_row + y + 1}"
            label_x = margin - 12
            label_y = page_height - margin - y * cell_size - cell_size / 2
            c.setFillColor(black)
            c.setStrokeColor(black)
            c.setLineWidth(0.5)
            c.rect(label_x - 1, label_y - 2, 10, 5, fill=0, stroke=1)
            c.drawString(label_x, label_y, label)

        # Column labels (C1, C2, ...)
        for x in range(len(grid[0])):
            label = f"C{x + 1}"
            label_x = margin + x * cell_size + 0.5
            label_y = page_height - margin + 4
            c.setFont("Helvetica", label_font_size)
            c.setFillColor(black)
            c.setStrokeColor(black)
            c.setLineWidth(0.5)
            c.rect(label_x - 1.5, label_y - 1, 9, 5, fill=0, stroke=1)
            c.drawString(label_x, label_y + 1, label)

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, margin / 2, f"Page {page_num + 1} of {num_pages}")
        c.showPage()

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, page_height - margin / 2, "Dice Count Summary")
    for i, (val, count) in enumerate(sorted(dice_count.items())):
        c.drawString(margin, page_height - margin - (i + 1) * 14, f"Dice {val}: {count}")

    c.save()
    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})