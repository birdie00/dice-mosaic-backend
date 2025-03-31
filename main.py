
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
from reportlab.lib.colors import black, white, red, blue, orange, green, yellow
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
    number_font_size = 5
    label_font_size = 3.5

    colors = {
        0: (black, white),
        1: (red, white),
        2: (blue, white),
        3: (green, white),  # swapped
        4: (orange, black), # swapped
        5: (yellow, black),
        6: (white, black),
    }

    dice_count = {i: 0 for i in range(0, 7)}

    rows = len(grid)
    cols = len(grid[0])
    half_rows = rows // 2
    num_pages = 2

    c = canvas.Canvas(filepath, pagesize=landscape(letter))

    for page in range(num_pages):
        start_row = page * half_rows
        end_row = min((page + 1) * half_rows, rows)

        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, page_height - margin / 2, "Pipcasso Dice Map")

        c.setFont("Helvetica", 10)
        c.drawString(margin, page_height - margin - 15, "Project: (Project Name)")
        c.drawString(margin, page_height - margin - 30, f"Dimensions: {cols} x {rows}")
        c.drawString(margin, page_height - margin - 45, "Instructions: Match each number with the corresponding dice face. Each square is a die. Use the row (R) and column (C) labels to place them correctly.")

        offset_y = 100
        grid_top = page_height - offset_y
        grid_bottom = margin + 40

        for y in range(start_row, end_row):
            for x in range(cols):
                val = grid[y][x]
                bg_color, text_color = colors.get(val, (white, black))
                px = margin + (x + 1) * cell_size
                py = grid_top - (y - start_row + 1) * cell_size

                c.setFillColor(bg_color)
                c.rect(px, py, cell_size, cell_size, fill=1, stroke=1)

                c.setFillColor(text_color)
                c.setFont("Helvetica", number_font_size)
                c.drawCentredString(px + cell_size / 2, py + 0.5, str(val))
                dice_count[val] += 1

        # Draw row labels
        for i in range(start_row, end_row):
            py = grid_top - (i - start_row + 1) * cell_size
            c.setFillColor(black)
            c.setFont("Helvetica", label_font_size)
            c.rect(margin, py, cell_size, cell_size, fill=0, stroke=1)
            c.drawCentredString(margin + cell_size / 2, py + 0.5, f"R{i + 1}")

        # Draw column labels
        for x in range(cols):
            px = margin + (x + 1) * cell_size
            py = grid_top
            c.setFillColor(black)
            c.setFont("Helvetica", label_font_size)
            c.rect(px, py, cell_size, cell_size, fill=0, stroke=1)
            c.drawCentredString(px + cell_size / 2, py + 0.5, f"C{x + 1}")

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, margin / 2, f"Page {page + 1} of {num_pages}")
        c.showPage()

    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, page_height - margin / 2, "Dice Count Summary")
    for i, (val, count) in enumerate(sorted(dice_count.items())):
        c.drawString(margin, page_height - margin - (i + 1) * 14, f"Dice {val}: {count}")

    c.save()
    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})
