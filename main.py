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

    dice_counts = {i: 0 for i in range(7)}
    for row in grid:
        for val in row:
            dice_counts[val] += 1

    # === Page 1: Info Page ===
    c.setFont("Helvetica-Bold", 22)
    title = "Pipcasso Dice Map"
    title_width = c.stringWidth(title, "Helvetica-Bold", 22)
    c.drawString((page_width - title_width) / 2, page_height - margin, title)

    c.setFont("Helvetica", 12)
    info_y = page_height - margin - 40
    c.drawString(margin, info_y, f"Project Name: {project_name}")
    c.drawString(margin, info_y - 20, f"Grid Size: {width} x {height}")

    # Instructions
    instructions_y = info_y - 60
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, instructions_y, "Instructions")
    instructions = [
        "1. Each number in the grid represents a dice face (0–6).",
        "2. Dice color is determined by number:",
        "   0: Black, 1: Red, 2: Blue, 3: Green, 4: Orange, 5: Yellow, 6: White",
        "3. The next page shows a preview of the full dice mosaic.",
        "4. Pages 3–6 break the mosaic into quadrants for easier building.",
        "5. Ghosted rows and columns help align quadrant edges.",
    ]
    c.setFont("Helvetica", 10)
    for i, line in enumerate(instructions):
        c.drawString(margin, instructions_y - 20 - (i * 14), line)

    # Dice Key
    key_y = instructions_y - 120
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, key_y, "Dice Map Key")

    col_widths = [60, 120, 60]
    headers = ["Color", "Dots (pips)", "Count"]
    c.setFont("Helvetica-Bold", 10)
    for i, header in enumerate(headers):
        c.drawString(margin + sum(col_widths[:i]) + 5, key_y + 20, header)

    c.setFont("Helvetica", 10)
    for i in range(7):
        y = key_y - 20 - (i * 18)
        r, g, b, _ = colors[i]
        c.setFillColorRGB(r / 255, g / 255, b / 255)
        c.rect(margin + 5, y + 4, 20, 10, fill=1, stroke=1)
        c.setFillColor(black)
        c.drawString(margin + col_widths[0] + 5, y, f"{i} face")
        c.drawString(margin + col_widths[0] + col_widths[1] + 5, y, str(dice_counts[i]))

    c.showPage()

    # === Page 2: Preview (smaller labels, quadrant outlines) ===
    preview_width = page_width - 2 * margin
    preview_height = page_height - 2 * margin - 20
    cell_size = min(preview_width / width, preview_height / height)
    c.saveState()
    c.translate(margin, margin)

    # Smaller preview with light text
    for y in range(height):
        for x in range(width):
            val = grid[y][x]
            r, g, b, _ = colors[val]
            c.setStrokeColorRGB(0.9, 0.9, 0.9)
            c.setLineWidth(0.3)
            c.setFillColorRGB(r / 255, g / 255, b / 255)
            px = x * cell_size
            py = (height - y - 1) * cell_size
            c.rect(px, py, cell_size, cell_size, fill=1, stroke=1)

    # Draw quadrant outlines (Problem 3)
    mid_x = width // 2
    mid_y = height // 2
    q_thickness = 1.5
    c.setStrokeColorRGB(0.3, 0.3, 0.3)
    c.setLineWidth(q_thickness)
    c.setDash(3, 2)
    quadrants = [
        (0, 0, mid_x + 1, mid_y + 1),
        (mid_x - 1, 0, width - mid_x + 1, mid_y + 1),
        (0, mid_y - 1, mid_x + 1, height - mid_y + 1),
        (mid_x - 1, mid_y - 1, width - mid_x + 1, height - mid_y + 1),
    ]
    for start_x, start_y, w, h in quadrants:
        px = start_x * cell_size
        py = (height - start_y - h) * cell_size
        c.rect(px, py, w * cell_size, h * cell_size, fill=0, stroke=1)
    c.setDash()  # reset dash

    c.restoreState()
    c.showPage()

    # === Pages 3–6: Quadrants ===
    for idx, (name, start_x, start_y, q_width, q_height) in enumerate(
        [("Top Left", 0, 0, mid_x + 1, mid_y + 1),
         ("Top Right", mid_x - 1, 0, width - mid_x + 1, mid_y + 1),
         ("Bottom Left", 0, mid_y - 1, mid_x + 1, height - mid_y + 1),
         ("Bottom Right", mid_x - 1, mid_y - 1, width - mid_x + 1, height - mid_y + 1)]
    ):
        c.setPageSize(pagesize)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, page_height - margin, f"Project: {project_name}")
        c.setFont("Helvetica", 14)
        c.drawString(margin, page_height - margin - 20, f"Quadrant: {name}")

        available_height = page_height - (margin + 80)
        available_width = page_width - 2 * margin
        cell_size = min(available_width / q_width, available_height / q_height)

        # Updated draw section for vertical centering + white border
        page_w, page_h = c._pagesize
        grid_total_w = cell_size * q_width
        grid_total_h = cell_size * q_height
        grid_left = (page_w - grid_total_w) / 2
        grid_top = (page_h + grid_total_h) / 2 - 40

        for y in range(q_height):
            for x in range(q_width):
                gx = start_x + x
                gy = start_y + y
                val = grid[gy][gx]
                r, g_, b, text_color = colors[val]

                px = grid_left + x * cell_size
                py = grid_top - y * cell_size

                # Background
                c.setFillColorRGB(r / 255, g_ / 255, b / 255)
                c.setStrokeColor(white)
                c.setLineWidth(0.3)
                c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)

                # Centered number
                c.setFillColor(text_color)
                c.setFont("Helvetica", 8)
                text_y_offset = c._fontsize / 2.7  # vertical centering fix (Problem 4)
                c.drawCentredString(
                    px + cell_size / 2,
                    py - cell_size / 2 - text_y_offset,
                    str(val)
                )

        c.showPage()

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
