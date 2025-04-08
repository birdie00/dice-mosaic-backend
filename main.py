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
    project_name: str  # ✅ add this line


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


from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape, portrait
from reportlab.lib.colors import black, white, gray
import os


def draw_grid_section(c, grid, start_x, start_y, width, height, cell_size, global_offset_x, global_offset_y,
                      colors, margin, label_font_size, number_font_size, ghost=False):
    """
    Draws a section of the grid starting at (start_x, start_y) with the given width and height.
    global_offset_* are used to label rows/cols with respect to the full grid.
    """
    page_width, page_height = c._pagesize
    grid_total_width = cell_size * width
    grid_total_height = cell_size * height
    grid_left = (page_width - grid_total_width) / 2
    grid_top = (page_height + grid_total_height) / 2

    for y in range(height):
        for x in range(width):
            val = grid[start_y + y][start_x + x]
            r, g, b, text_color = colors[val]
            px = grid_left + x * cell_size
            py = grid_top - y * cell_size

            if ghost and (x == width - 1 or y == height - 1):
                c.setFillColor(gray)
            else:
                c.setFillColorRGB(r / 255, g / 255, b / 255)

            c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=0)

            if ghost and (x == width - 1 or y == height - 1):
                c.setFillColor(gray)
            else:
                c.setFillColor(text_color)
            c.setFont("Helvetica", number_font_size)
            c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - (number_font_size / 2) * 0.3, str(val))

    # Row/Col Labels
    for x in range(width):
        label = f"C{start_x + x + 1}"
        px = grid_left + x * cell_size
        py = grid_top + cell_size
        c.setFillColor(white)
        c.setStrokeColor(black)
        c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", label_font_size)
        c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - (label_font_size / 2) * 0.3, label)

    for y in range(height):
        label = f"R{start_y + y + 1}"
        px = grid_left - cell_size
        py = grid_top - y * cell_size
        c.setFillColor(white)
        c.setStrokeColor(black)
        c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)
        c.setFillColor(black)
        c.setFont("Helvetica", label_font_size)
        c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - (label_font_size / 2) * 0.3, label)


# This function will be called inside your FastAPI route

def generate_better_dice_pdf(filepath, grid, project_name):
    height = len(grid)
    width = len(grid[0])
    is_portrait = height >= width
    pagesize = portrait(letter) if is_portrait else landscape(letter)

    cell_size = 6
    margin = 40
    label_font_size = 2.3
    number_font_size = 3.5

    colors = {
        0: (0, 0, 0, white),
        1: (255, 0, 0, white),
        2: (0, 0, 255, white),
        3: (0, 128, 0, white),
        4: (255, 165, 0, black),
        5: (255, 255, 0, black),
        6: (255, 255, 255, black),
    }

    c = canvas.Canvas(filepath, pagesize=pagesize)

    # --- Page 1: Overview + Instructions ---
    page_width, page_height = pagesize
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
    for i, line in enumerate(instructions):
        c.drawString(margin, page_height - margin - 80 - (i * 16), line)

    # Mini Grid
    mini_cell = 1  # small cell size
    draw_grid_section(c, grid, 0, 0, width, height, mini_cell, 0, 0, colors,
                      margin, 1.5, 2.0, ghost=False)
    c.showPage()

    # --- Pages 2–5: Quadrants ---
    mid_x = width // 2
    mid_y = height // 2

    quadrants = [
        (0, 0, mid_x + 1, mid_y + 1),  # top-left with right/bottom ghost
        (mid_x - 1, 0, width - mid_x + 1, mid_y + 1),  # top-right with left/bottom ghost
        (0, mid_y - 1, mid_x + 1, height - mid_y + 1),  # bottom-left with top/right ghost
        (mid_x - 1, mid_y - 1, width - mid_x + 1, height - mid_y + 1),  # bottom-right with top/left ghost
    ]

    for start_x, start_y, quad_width, quad_height in quadrants:
        c.setPageSize(pagesize)
        draw_grid_section(
            c, grid,
            start_x, start_y,
            quad_width, quad_height,
            cell_size, start_x, start_y,
            colors, margin, label_font_size, number_font_size,
            ghost=True
        )
        c.showPage()

    c.save()

@app.post("/generate-pdf")
async def generate_dice_map_pdf(grid_data: GridRequest):
    grid = grid_data.grid_data
    project_name = grid_data.project_name  # Keep this!
    filename = f"dice_map_{uuid4().hex}.pdf"
    filepath = os.path.join("static", filename)

    generate_better_dice_pdf(filepath, grid, project_name)

    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})
