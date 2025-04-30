from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from uuid import uuid4
from PIL import Image, ImageEnhance, ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape, portrait
from reportlab.lib.colors import black, white, gray
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors as rl_colors
from reportlab.lib.units import mm
import numpy as np
import os
import cv2


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TEMP fix: open to all, including vercel
    allow_credentials=False,        # IMPORTANT: must be False when using "*"
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
    print(f"[DEBUG] /analyze received: grid_width={grid_width}, grid_height={grid_height}")

    original = Image.open(file.file).convert("L")
    base = original.resize((grid_width, grid_height))

    print(f"[DEBUG] Resized image to: {base.size}")  # <-- Confirm actual size

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
        processed = apply_enhancements(base.copy(), **settings)
        arr = np.array(processed)
        print(f"[DEBUG] Style {style_id} -> numpy shape: {arr.shape}")  # <-- Check grid shape

        grid = [[int(val / 256 * 7) for val in row] for row in arr.tolist()]
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
            c.setStrokeColor(white)
            c.setLineWidth(0.5)
            c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)
            c.setFillColor(gray if is_ghost_cell else text_color)
            c.setFont("Helvetica", number_font_size)
            text_y = py - cell_size / 2 - (number_font_size / 2) * 0.3
            c.drawCentredString(px + cell_size / 2, text_y, str(val))

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
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import landscape, letter, portrait
    from reportlab.lib.colors import Color, black, white, lightgrey
    from reportlab.lib.units import inch

    rows, cols = len(grid), len(grid[0])
    use_portrait = cols > 100
    page_size = portrait(letter) if use_portrait else landscape(letter)
    page_width, page_height = page_size
    margin = 0.25 * inch
    c = canvas.Canvas(filepath, pagesize=page_size)

    # Color mapping
    color_map = {
        0: (Color(0, 0, 0), white),
        1: (Color(1, 0, 0), white),
        2: (Color(0, 0, 1), white),
        3: (Color(1, 0.55, 0), black),
        4: (Color(0, 0.5, 0), white),
        5: (Color(1, 1, 0), black),
        6: (Color(1, 1, 1), black),
    }

    # Layout calculations
    cell_size = min((page_width - 2 * margin) / (cols + 1), (page_height - 2.25 * inch) / ((rows // 2) + 2), 10)
    grid_start_x = margin + cell_size
    label_font_size = 3.5
    number_font_size = 5.2  # reduced for better fit

    def draw_header_and_key():
        text_x = margin
        key_x = page_width * 0.6
        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(page_width / 2, page_height - margin, f"Pipcasso Dice Map for '{project_name}'")
        c.setFont("Helvetica", 10)
        top = page_height - margin - 20
        c.drawString(text_x, top, f"Project: {project_name}")
        c.drawString(text_x, top - 15, f"Dimensions: {cols} W x {rows} H")
        c.drawString(text_x, top - 30, "Instructions: Match the numbers on this blueprint to the dice faces.")
        c.drawString(text_x, top - 45, "Blank (0 Face) dice can be made by coloring a '1' face with a black marker.")

        # Key on the right
        y = top
        row_height = 12
        col_widths = [50, 35, 50]
        headers = ["Color", "Dice", "Count"]
        c.setFont("Helvetica-Bold", 9)
        x = key_x
        for i, header in enumerate(headers):
            c.setFillColor(lightgrey)
            c.rect(x, y - row_height, col_widths[i], row_height, stroke=1, fill=1)
            c.setFillColor(black)
            c.drawCentredString(x + col_widths[i]/2, y - row_height + 3, header)
            x += col_widths[i]

        y -= row_height
        c.setFont("Helvetica", 8)
        for i in range(7):
            x = key_x
            bg, _ = color_map[i]
            label = ["Black", "Red", "Blue", "Orange", "Green", "Yellow", "White"][i]
            count = sum(row.count(i) for row in grid)
            entries = [label, f"{i} face", f"{count}"]
            for j, text in enumerate(entries):
                c.setFillColor(bg if j == 0 else white)
                c.rect(x, y - row_height, col_widths[j], row_height, stroke=1, fill=1)
                text_color = black if (i in [5, 6] and j == 0) else (white if j == 0 else black)
                c.setFillColor(text_color)
                c.drawCentredString(x + col_widths[j]/2, y - row_height + 3, text)
                x += col_widths[j]
            y -= row_height
        return y - 10

    def draw_column_headers(start_y):
        c.setFont("Helvetica", label_font_size)
        for col in range(cols):
            x = grid_start_x + col * cell_size
            c.setFillColor(lightgrey)
            c.setStrokeColor(black)
            c.rect(x, start_y, cell_size, cell_size, fill=1, stroke=1)
            c.setFillColor(black)
            c.drawCentredString(x + cell_size / 2, start_y + cell_size * 0.3, f"C{col+1}")

    def draw_grid(start_row, end_row, start_y):
        draw_column_headers(start_y)
        y_origin = start_y - cell_size
        for row_idx in range(start_row, end_row):
            y = y_origin - ((row_idx - start_row) * cell_size)
            c.setFillColor(lightgrey)
            c.setStrokeColor(black)
            c.rect(margin, y, cell_size, cell_size, fill=1, stroke=1)
            c.setFillColor(black)
            c.setFont("Helvetica", label_font_size)
            c.drawCentredString(margin + cell_size / 2, y + cell_size * 0.3, f"R{row_idx+1}")

            for col in range(cols):
                val = grid[row_idx][col]
                bg, fg = color_map[val]
                x = grid_start_x + col * cell_size
                c.setFillColor(bg)
                c.rect(x, y, cell_size, cell_size, fill=1, stroke=0)
                c.setStrokeColor(white)
                c.setLineWidth(0.5)
                c.rect(x, y, cell_size, cell_size, fill=0, stroke=1)
                c.setFillColor(fg)
                c.setFont("Helvetica", number_font_size)
                c.drawCentredString(x + cell_size / 2, y + cell_size * 0.3, str(val))

    # Page 1
    bottom_y = draw_header_and_key()
    half = rows // 2
    draw_grid(0, half, bottom_y)
    c.showPage()

    # Page 2
    draw_grid(half, rows, page_height - margin)
    c.save()

@app.post("/generate-pdf")
async def generate_dice_map_pdf(grid_data: GridRequest):
    grid = grid_data.grid_data
    actual_height = len(grid)
    actual_width = len(grid[0]) if actual_height > 0 else 0
    print(f"[DEBUG] PDF generation: received grid size = {actual_width} cols x {actual_height} rows")
    project_name = grid_data.project_name
    filename = f"dice_map_{uuid4().hex}.pdf"
    filepath = os.path.join("static", filename)

    generate_better_dice_pdf(filepath, grid, project_name)

    return JSONResponse(content={"dice_map_url": f"/static/{filename}"})
from fastapi import Request
from PIL import ImageDraw

@app.post("/generate-image")
async def generate_image(request: Request):
    print("🎯 /generate-image hit")
    body = await request.json()
    grid = body.get("grid_data")
    style_id = body.get("style_id")
    project_name = body.get("project_name", "Pipcasso")
    resolution = body.get("resolution", "low")
    print("🧩 Request resolution:", resolution)

    mode = body.get("mode", "dice")

    if not grid:
        return JSONResponse(status_code=400, content={"error": "Missing grid_data"})

    # Set dice image folder inside backend
    dice_dir = os.path.join(os.getcwd(), "dice")
    print("🧾 dice_dir:", dice_dir)
    print("📂 dice_dir contents:", os.listdir(dice_dir) if os.path.exists(dice_dir) else "MISSING")

    # Load and resize dice images
    dice_size = 20 if resolution == "low" else 75

    try:
        dice_images = {
            i: Image.open(os.path.join(dice_dir, f"dice_{i}.png")).convert("RGBA").resize((dice_size, dice_size), Image.LANCZOS)
            for i in range(7)
        }
    except Exception as e:
        print(f"❌ Error loading dice images: {e}")
        return JSONResponse(status_code=500, content={"error": "Server failed to load dice images."})

    height = len(grid)
    width = len(grid[0])
    img_width = width * dice_size
    img_height = height * dice_size

    mosaic = Image.new("RGBA", (img_width, img_height), (255, 255, 255, 255))

    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            dice_val = int(val)
            if dice_val in dice_images:
                dice_img = dice_images[dice_val]
                mosaic.paste(dice_img, (x * dice_size, y * dice_size), mask=dice_img)

    filename = f"dice_mosaic_{resolution}_{uuid4().hex}.png"
    filepath = os.path.join("static", filename)
    mosaic.convert("RGB").save(filepath)

    return JSONResponse(content={"image_url": f"/static/{filename}"})
