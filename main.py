from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from uuid import uuid4
from PIL import Image, ImageEnhance
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape, portrait
from reportlab.lib.colors import black, white, gray
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors as rl_colors
from PIL import Image
import numpy as np
import os
import cv2
from fastapi import Request
from PIL import ImageDraw


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



def generate_better_dice_pdf(filepath, grid, project_name, dice_dir):
    height = len(grid)
    width = len(grid[0])
    is_portrait = height >= width
    pagesize = portrait(letter) if is_portrait else landscape(letter)
    c = canvas.Canvas(filepath, pagesize=pagesize)
    page_width, page_height = pagesize
    margin = 40

    # Define dice color key (for table coloring)
    colors = {
        0: (0, 0, 0, rl_colors.white),
        1: (255, 0, 0, rl_colors.white),
        2: (0, 0, 255, rl_colors.white),
        3: (0, 128, 0, rl_colors.white),
        4: (255, 165, 0, rl_colors.black),
        5: (255, 255, 0, rl_colors.black),
        6: (255, 255, 255, rl_colors.black),
    }

    # Calculate dice counts
    dice_counts = {i: 0 for i in range(7)}
    for row in grid:
        for val in row:
            dice_counts[val] += 1

    mid_x = width // 2
    mid_y = height // 2

    # === PAGE 1 ===
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(page_width / 2, page_height - margin, "Pipcasso Dice Map")

    section_y = page_height - margin - 40
    top_left_x = margin

    # Project Info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(top_left_x, section_y, "Project Info")
    c.setFont("Helvetica", 11)
    c.drawString(top_left_x, section_y - 20, f"Project Name: {project_name}")
    c.drawString(top_left_x, section_y - 40, f"Grid Size: {width} x {height}")

    # Instructions
    c.setFont("Helvetica-Bold", 12)
    c.drawString(top_left_x, section_y - 70, "Instructions")
    c.setFont("Helvetica", 10)
    instructions = [
        "• Each number represents a dice face (0–6).",
        "• Build quadrant-by-quadrant for easier assembly.",
        "• Pages 2–5 contain quadrant instructions.",
    ]
    for i, line in enumerate(instructions):
        c.drawString(top_left_x, section_y - 90 - (i * 14), line)

    # Draw Dice Preview
    preview_size = 8  # size of each tiny dice image
    preview_grid_width = width * preview_size
    preview_grid_height = height * preview_size
    preview_left = (page_width - preview_grid_width) / 2
    preview_bottom = margin + 180  # enough space above dice table

    # Load dice images
    dice_images = {}
    for i in range(7):
        dice_path = os.path.join(dice_dir, f"dice_{i}.png")
        if os.path.exists(dice_path):
            dice_images[i] = Image.open(dice_path).resize((preview_size, preview_size))

    for y, row in enumerate(grid):
        for x, val in enumerate(row):
            if val in dice_images:
                dice_img_path = os.path.join(dice_dir, f"dice_{val}.png")
                c.drawImage(
                    dice_img_path,
                    preview_left + x * preview_size,
                    preview_bottom + (height - y - 1) * preview_size,
                    width=preview_size,
                    height=preview_size,
                    preserveAspectRatio=True,
                    mask='auto'
                )

    # === Dice Map Key Table ===
    data = [["Color", "Dice Face", "Count"]]
    for i in range(7):
        data.append(["", f"{i}", str(dice_counts[i])])

    dice_key_table = Table(data, colWidths=[40, 80, 50])
    dice_key_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, rl_colors.black),
        ('BACKGROUND', (0,0), (-1,0), rl_colors.lightgrey),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,0), 10),
        ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,1), (-1,-1), 9),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))

    for i in range(1, 8):
        r, g, b, _ = colors[i-1]
        dice_key_table.setStyle([
            ('BACKGROUND', (0,i), (0,i), rl_colors.Color(r/255, g/255, b/255))
        ])

    w, h = dice_key_table.wrapOn(c, page_width, page_height)
    dice_key_table.drawOn(c, page_width - w - margin, preview_bottom - h - 20)

    c.showPage()

    # Quadrants (Pages 2-5)
    quadrants = [
        ("Top Left", 0, 0, mid_x + 1, mid_y + 1),
        ("Top Right", mid_x - 1, 0, width - mid_x + 1, mid_y + 1),
        ("Bottom Left", 0, mid_y - 1, mid_x + 1, height - mid_y + 1),
        ("Bottom Right", mid_x - 1, mid_y - 1, width - mid_x + 1, height - mid_y + 1),
    ]
    for name, start_x, start_y, q_width, q_height in quadrants:
        c.setPageSize(pagesize)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(margin, page_height - margin, f"Project: {project_name}")
        c.setFont("Helvetica", 14)
        c.drawString(margin, page_height - margin - 20, f"Quadrant: {name}")

        available_height = page_height - (margin + 80)
        available_width = page_width - 2 * margin
        cell_size = min(available_width / (q_width + 1), available_height / (q_height + 1))

        grid_left = (page_width - cell_size * (q_width + 1)) / 2
        grid_top = (page_height + cell_size * (q_height + 1)) / 2 - 40

        for y in range(q_height):
            for x in range(q_width):
                gx = start_x + x
                gy = start_y + y
                val = grid[gy][gx]
                r, g_, b, text_color = colors[val]
                px = grid_left + (x + 1) * cell_size
                py = grid_top - (y + 1) * cell_size
                c.setFillColorRGB(r / 255, g_ / 255, b / 255)
                c.setStrokeColor(white)
                c.setLineWidth(0.3)
                c.rect(px, py, cell_size, cell_size, fill=1, stroke=1)
                c.setFont("Helvetica", 8)
                text_offset = c._fontsize / 2.5
                c.setFillColor(text_color)
                c.drawCentredString(px + cell_size / 2, py + cell_size / 2 - text_offset, str(val))

        # Column and Row Labels
        c.setFont("Helvetica", 8)
        for x in range(q_width):
            label = f"C{start_x + x + 1}"
            px = grid_left + (x + 1) * cell_size
            py = grid_top
            c.drawCentredString(px + cell_size / 2, py + 2, label)
        for y in range(q_height):
            label = f"R{start_y + y + 1}"
            px = grid_left
            py = grid_top - (y + 1) * cell_size
            c.drawCentredString(px + cell_size / 2, py + cell_size / 2 - 3, label)

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
