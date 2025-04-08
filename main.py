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
from reportlab.lib.colors import black, white, gray, red, lightgrey
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
    project_name: str


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
        3: {"brightness": 1.3, "contrast": 1.5, "sharpness": 1.4, "clahe": True, "gamma": 0.85},
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


# FIX: Limit grid drawing area to avoid overlapping instructions

def generate_better_dice_pdf(filepath, grid, project_name):
    height = len(grid)
    width = len(grid[0])
    is_portrait = height >= width
    pagesize = portrait(letter) if is_portrait else landscape(letter)

    c = canvas.Canvas(filepath, pagesize=pagesize)
    page_width, page_height = pagesize
    margin = 40

    colors = {
        0: (0, 0, 0, white),
        1: (255, 0, 0, white),
        2: (0, 0, 255, white),
        3: (0, 128, 0, white),
        4: (255, 165, 0, black),
        5: (255, 255, 0, black),
        6: (255, 255, 255, black),
    }

    # Page 1
    c.setFont("Helvetica-Bold", 22)
    c.drawString(margin, page_height - margin, "Pipcasso Dice Map")
    c.setFont("Helvetica", 12)
    c.drawString(margin, page_height - margin - 30, f"Project Name: {project_name}")
    c.drawString(margin, page_height - margin - 50, f"Grid Size: {width} x {height}")
    instructions = [
        "Instructions:",
        "1. Each number in the grid represents a dice face (0–6).",
        "2. Dice color is determined by number:",
        "   0: Black, 1: Red, 2: Blue, 3: Green, 4: Orange, 5: Yellow, 6: White",
        "3. Use quadrant pages to place dice in sections.",
        "4. Ghosted rows/columns help you align your sections correctly.",
    ]
    c.setFont("Helvetica", 10)
    y_start = page_height - margin - 80
    for i, line in enumerate(instructions):
        c.drawString(margin, y_start - (i * 16), line)

    instructions_height = (len(instructions) * 16) + 100  # 80 top spacing + 20 buffer
    grid_start_y = page_height - margin - instructions_height
    grid_area_height = grid_start_y - margin
    grid_area_width = page_width - 2 * margin
    cell_size = min(grid_area_width / width, grid_area_height / height)

    from_page_bottom = margin
    draw_grid_section(c, grid, 0, 0, width, height, cell_size, 0, 0, colors,
                      margin, 1.5, 2.0, ghost=False)
    c.showPage()

    mid_x = width // 2
    mid_y = height // 2
    quadrants = [
        ("Top Left", 0, 0, mid_x + 1, mid_y + 1),
        ("Top Right", mid_x - 1, 0, width - mid_x + 1, mid_y + 1),
        ("Bottom Left", 0, mid_y - 1, mid_x + 1, height - mid_y + 1),
        ("Bottom Right", mid_x - 1, mid_y - 1, width - mid_x + 1, height - mid_y + 1),
    ]

    for quadrant_name, start_x, start_y, quad_width, quad_height in quadrants:
        c.setPageSize(pagesize)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, page_height - margin, f"Project: {project_name}")
        c.setFont("Helvetica", 14)
        c.drawString(margin, page_height - margin - 20, f"Quadrant: {quadrant_name}")

        draw_section_preview(
            c, width, height,
            start_x, start_y,
            quad_width, quad_height,
            page_width - margin - 80,
            page_height - margin - 10
        )

        available_height = page_height - (margin + 80)
        available_width = page_width - 2 * margin
        cell_size = min(available_width / quad_width, available_height / quad_height)

        draw_grid_section(
            c, grid,
            start_x, start_y,
            quad_width, quad_height,
            cell_size, start_x, start_y,
            colors, margin, 2.5, 4.5,
            ghost=True
        )
        c.showPage()

    c.save()


@app.post("/generate-pdf")
async def generate_dice_map_pdf(grid_data: GridRequest):
    grid = grid_data.grid_data
    project_name = grid_data.project_name
    filename = f"dice_map_{uuid4().hex}.pdf"
    filepath = os.path.join("static", filename)

    generate_better_dice_pdf(filepath, grid, project_name)

    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})