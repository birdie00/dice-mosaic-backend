
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from uuid import uuid4
from PIL import Image, ImageEnhance
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, portrait, letter
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
    style_id: int


def apply_enhancements(pil_img, brightness, contrast, sharpness, gamma=1.0, clahe=False):
    if clahe:
        cv_img = np.array(pil_img)
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cv_img = clahe_op.apply(cv_img)
        pil_img = Image.fromarray(cv_img)

    pil_img = ImageEnhance.Brightness(pil_img).enhance(brightness)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast)
    pil_img = ImageEnhance.Sharpness(pil_img).enhance(sharpness)

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
    style_id = grid_data.style_id
    filename = f"dice_map_{uuid4().hex}.pdf"
    filepath = os.path.join("static", filename)

    height = len(grid)
    width = len(grid[0])
    is_portrait = height > width

    pagesize = portrait(letter) if is_portrait else landscape(letter)
    page_width, page_height = pagesize
    cell_size = 6
    margin = 40
    font_size = 2.3
    label_font_size = 2.3
    number_font_size = 3.5
    preview_scale = 3

    colors = {
        0: (0, 0, 0, white),
        1: (255, 0, 0, white),
        2: (0, 0, 255, white),
        3: (0, 128, 0, white),
        4: (255, 165, 0, black),
        5: (255, 255, 0, black),
        6: (255, 255, 255, black),
    }

    def draw_preview(c, x, y):
        for j, row in enumerate(grid[:20]):
            for i, val in enumerate(row[:20]):
                if val not in colors:
                    continue
                r, g, b, _ = colors[val]
                px = x + i * preview_scale
                py = y - j * preview_scale
                c.setFillColorRGB(r / 255, g / 255, b / 255)
                c.rect(px, py - preview_scale, preview_scale, preview_scale, fill=1, stroke=0)
                c.setFillColor(black)
                c.setFont("Helvetica", 1)
                c.drawCentredString(px + preview_scale / 2, py - preview_scale + 0.2, str(val))

    c = canvas.Canvas(filepath, pagesize=pagesize)

    # Page 1 — Title, instructions, and preview
    c.setFont("Helvetica-Bold", 22)
    c.drawString(margin, page_height - margin, "Pipcasso Dice Map")
    c.setFont("Helvetica", 12)
    c.drawString(margin, page_height - margin - 30, f"Project Name: (To Be Filled)")
    c.drawString(margin, page_height - margin - 50, f"Grid Size: {width} x {height}")

    instructions = [
        "Instructions:",
        "1. Each number in the grid represents a dice face (0–6).",
        "2. Dice color is determined by number:",
        "   0: Black, 1: Red, 2: Blue, 3: Green, 4: Orange, 5: Yellow, 6: White",
        "3. Match the dice number and position to recreate the image.",
        "4. Use row (R) and column (C) labels to help place dice accurately.",
    ]

    c.setFont("Helvetica", 10)
    for i, line in enumerate(instructions):
        c.drawString(margin, page_height - margin - 80 - (i * 16), line)

    # Move preview down
    preview_x = margin
    preview_y = page_height - margin - 220
    draw_preview(c, preview_x, preview_y)

    c.showPage()

    # Page 2 — Dice Map Only
    c.setPageSize(pagesize)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(page_width / 2, page_height - margin, "Pipcasso Dice Map")

    grid_top = page_height - margin - 20
    grid_left = margin + cell_size

    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            r, g, b, text_color = colors[val]
            px = grid_left + x * cell_size
            py = grid_top - y * cell_size

            c.setFillColorRGB(r / 255, g / 255, b / 255)
            c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=0)

            c.setFillColor(white)
            c.rect(px, py - cell_size, cell_size, cell_size, fill=0, stroke=1)

            c.setFillColor(text_color)
            c.setFont("Helvetica", number_font_size)
            c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - 1, str(val))

    # Column headers
    for x in range(width):
        label = f"C{x + 1}"
        px = grid_left + x * cell_size
        py = grid_top + cell_size
        c.setFillColor(white)
        c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", label_font_size)
        c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - 1, label)

    # Row headers
    for y in range(height):
        label = f"R{y + 1}"
        px = margin
        py = grid_top - y * cell_size
        c.setFillColor(white)
        c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", label_font_size)
        c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - 1, label)

    c.save()
    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})
