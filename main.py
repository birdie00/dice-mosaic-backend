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
from reportlab.lib.colors import black, white, gray
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


def draw_grid_section(c, grid, start_x, start_y, width, height, cell_size, global_offset_x, global_offset_y,
                      colors, margin, label_font_size, number_font_size, ghost=False):
    page_width, page_height = c._pagesize
    grid_total_width = cell_size * width
    grid_total_height = cell_size * height
    grid_left = (page_width - grid_total_width) / 2
    grid_top = (page_height + grid_total_height) / 2 - 40

    for y in range(height):
        for x in range(width):
            val = grid[start_y + y][start_x + x]
            r, g, b, text_color = colors[val]
            px = grid_left + x * cell_size
            py = grid_top - y * cell_size
            is_ghost_cell = ghost and (x == width - 1 or y == height - 1)
            c.setFillColor(gray if is_ghost_cell else (r / 255, g / 255, b / 255))
            c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=0)
            c.setFillColor(gray if is_ghost_cell else text_color)
            c.setFont("Helvetica", number_font_size + 2)
            c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - ((number_font_size + 2) / 2) * 0.3, str(val))

    for x in range(width):
        label = f"C{start_x + x + 1}"
        px = grid_left + x * cell_size
        py = grid_top + cell_size
        is_ghost_label = ghost and x == width - 1
        c.setFillColor(white)
        c.setStrokeColor(gray if is_ghost_label else black)
        c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)
        c.setFillColor(gray if is_ghost_label else black)
        c.setFont("Helvetica", label_font_size)
        c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - (label_font_size / 2) * 0.3, label)

    for y in range(height):
        label = f"R{start_y + y + 1}"
        px = grid_left - cell_size
        py = grid_top - y * cell_size
        is_ghost_label = ghost and y == height - 1
        c.setFillColor(white)
        c.setStrokeColor(gray if is_ghost_label else black)
        c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)
        c.setFillColor(gray if is_ghost_label else black)
        c.setFont("Helvetica", label_font_size)
        c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - (label_font_size / 2) * 0.3, label)


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
        3: (255, 165, 0, black),
        4: (0, 128, 0, white),
        5: (255, 255, 0, black),
        6: (255, 255, 255, black),
    }

    # Page 1: Top Half
    c.setFont("Helvetica-Bold", 22)
    title = "Pipcasso Dice Map"
    title_width = c.stringWidth(title, "Helvetica-Bold", 22)
    c.drawString((page_width - title_width) / 2, page_height - margin, title)

    # Project details and instructions (left)
    left_x = margin
    left_y = page_height - margin - 40
    c.setFont("Helvetica", 12)
    c.drawString(left_x, left_y, f"Project Name: {project_name}")
    c.drawString(left_x, left_y - 20, f"Grid Size: {width} x {height}")

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
        c.drawString(left_x, left_y - 50 - (i * 14), line)

    # Dice counts (right side key)
    dice_counts = {i: 0 for i in range(7)}
    for row in grid:
        for val in row:
            dice_counts[val] += 1

    right_x = page_width / 2 + 40
    right_y = page_height - margin - 40
    c.setFont("Helvetica-Bold", 12)
    c.drawString(right_x, right_y, "Key")
    c.setFont("Helvetica-Bold", 10)
    c.drawString(right_x, right_y - 16, "Colour")
    c.drawString(right_x + 60, right_y - 16, "Dice")
    c.drawString(right_x + 110, right_y - 16, "Count")

    c.setFont("Helvetica", 10)
    for i in range(7):
        y = right_y - 32 - (i * 14)
        r, g, b, text_color = colors[i]
        c.setFillColorRGB(r / 255, g / 255, b / 255)
        c.rect(right_x, y, 20, 10, fill=1, stroke=1)

        c.setFillColor(text_color)
        c.drawString(right_x + 25, y, f"{i} face")
        c.setFillColor(black)
        c.drawString(right_x + 75, y, str(dice_counts[i]))

    # Grid Preview (bottom half)
    preview_height = page_height / 2 - margin
    preview_width = page_width - 2 * margin
    cell_size = min(preview_width / width, preview_height / height)
    grid_height_in_points = cell_size * height
    grid_y_center = (page_height / 2 - grid_height_in_points) / 2

    c.saveState()
    c.translate(0, grid_y_center)
    draw_grid_section(c, grid, 0, 0, width, height, cell_size, 0, 0, colors, margin, 1.5, 2.0, ghost=False)
    c.restoreState()
    c.showPage()

    # Quadrants
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
