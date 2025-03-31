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
from reportlab.lib.colors import black, white, red, blue, green, yellow, orange
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
    font_size = 4  # for numbers inside blocks
    label_font_size = int(font_size * 0.75)  # for row/column headers

    color_styles = {
        0: {"fill": black, "text": white},
        1: {"fill": red, "text": white},
        2: {"fill": blue, "text": white},
        3: {"fill": orange, "text": black},
        4: {"fill": green, "text": black},
        5: {"fill": yellow, "text": black},
        6: {"fill": white, "text": black},
    }

    total_rows = len(grid)
    total_cols = len(grid[0]) if total_rows > 0 else 0
    half_rows = total_rows // 2

    c = canvas.Canvas(filepath, pagesize=landscape(letter))

    for page_num in range(2):
        start_row = page_num * half_rows
        end_row = min(start_row + half_rows, total_rows)

        # HEADER INFO
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, page_height - margin / 2, "Pipcasso Dice Map")

        c.setFont("Helvetica", 10)
        c.drawString(margin, page_height - margin - 12, f"Project: [Add Name]")
        c.drawString(margin, page_height - margin - 26, f"Dimensions: {total_cols} x {total_rows}")
        c.drawString(margin, page_height - margin - 40, "Instructions: Place dice following color and number in grid. Match R/C labels.")

        top_offset = page_height - margin - 60
        left_offset = margin + cell_size

        # DRAW COLUMN HEADINGS
        for col in range(total_cols):
            x = left_offset + col * cell_size
            c.setFillColor(white)
            c.rect(x, top_offset, cell_size, cell_size, fill=1, stroke=1)
            c.setFillColor(black)
            c.setFont("Helvetica", label_font_size)
            c.drawCentredString(x + cell_size / 2, top_offset + 1.5, f"C{col+1}")

        # DRAW GRID
        for row_idx, row in enumerate(grid[start_row:end_row]):
            y = top_offset - (row_idx + 1) * cell_size

            # Row label
            c.setFillColor(white)
            c.rect(margin, y, cell_size, cell_size, fill=1, stroke=1)
            c.setFillColor(black)
            c.setFont("Helvetica", label_font_size)
            c.drawCentredString(margin + cell_size / 2, y + 1.5, f"R{start_row + row_idx + 1}")

            for col_idx, val in enumerate(row):
                color = color_styles.get(val, {"fill": white, "text": black})
                x = left_offset + col_idx * cell_size
                c.setFillColor(color["fill"])
                c.rect(x, y, cell_size, cell_size, fill=1, stroke=1)
                c.setFillColor(color["text"])
                c.setFont("Helvetica", font_size)
                c.drawCentredString(x + cell_size / 2, y + 1.5, str(val))

        c.setFont("Helvetica-Bold", 10)
        c.drawString(margin, margin / 2, f"Page {page_num + 1} of 2")
        c.showPage()

    c.save()
    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})