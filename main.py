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
from reportlab.lib.colors import black, white, HexColor
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
    grid_width = len(grid[0])
    grid_height = len(grid)

    text_colors = {0: white, 1: white, 2: white, 3: black, 4: black, 5: black, 6: black}
    fill_colors = {
        0: black,
        1: HexColor("#FF4B4B"),
        2: HexColor("#4B7BFF"),
        3: HexColor("#FFA500"),
        4: HexColor("#3CB371"),
        5: HexColor("#FFFF66"),
        6: white
    }

    rows_per_page = grid_height // 2
    num_pages = 2

    c = canvas.Canvas(filepath, pagesize=landscape(letter))

    for page_num in range(num_pages):
        start_row = page_num * rows_per_page
        end_row = min(start_row + rows_per_page, grid_height)

        # Title and instructions
        c.setFont("Helvetica-Bold", 14)
        c.drawCentredString(page_width / 2, page_height - 30, "Pipcasso Dice Map")
        c.setFont("Helvetica", 10)
        c.drawString(margin, page_height - 45, f"Project: [Insert Name Here]")
        c.drawString(margin, page_height - 60, f"Grid Size: {grid_width} x {grid_height}")
        c.drawString(margin, page_height - 75, "Instructions: Match the dice number in each block.")

        offset_y = page_height - margin - 100

        for y, row in enumerate(grid[start_row:end_row]):
            row_index = start_row + y
            for x, val in enumerate(row):
                px = margin + (x + 1) * cell_size
                py = offset_y - y * cell_size

                c.setStrokeColor(white)
                c.setLineWidth(0.2)
                c.setFillColor(fill_colors[val])
                c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)

                c.setFillColor(text_colors[val])
                c.setFont("Helvetica", font_size)
                c.drawCentredString(px + cell_size / 2, py - cell_size + 1.5, str(val))

        for y in range(start_row, end_row):
            row_y = offset_y - (y - start_row) * cell_size
            c.setStrokeColor(black)
            c.setFillColor(black)
            c.setFont("Helvetica", 3.5)
            c.rect(margin, row_y - cell_size, cell_size, cell_size, fill=0)
            c.drawCentredString(margin + cell_size / 2, row_y - cell_size + 1.5, f"R{y+1}")

        for x in range(grid_width):
            col_x = margin + (x + 1) * cell_size
            c.setStrokeColor(black)
            c.setFillColor(black)
            c.setFont("Helvetica", 3.5)
            c.rect(col_x, offset_y + cell_size, cell_size, cell_size, fill=0)
            c.drawCentredString(col_x + cell_size / 2, offset_y + cell_size + 1.5, f"C{x+1}")

        c.showPage()

    c.save()
    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})