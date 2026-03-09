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
    allow_origins=[
        "https://www.pipcasso.com",
        "https://pipcasso.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
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
    if grid_width < 10 or grid_height < 10 or grid_width > 1000 or grid_height > 1000:
        return JSONResponse(
            status_code=400,
            content={"error": "Grid size out of range. Must be between 10×10 and 1000×1000."}
        )

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
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.colors import Color, black, white, lightgrey, darkgrey
    from reportlab.lib.units import inch

    rows, cols = len(grid), len(grid[0])
    pw, ph = landscape(letter)   # 792 x 612 pts
    margin = 0.25 * inch         # 18 pts

    color_map = {
        0: (Color(0, 0, 0),        white),
        1: (Color(1, 0, 0),        white),
        2: (Color(0, 0, 1),        white),
        3: (Color(1, 0.55, 0),     black),
        4: (Color(0, 0.5, 0),      white),
        5: (Color(1, 1, 0),        black),
        6: (Color(1, 1, 1),        black),
    }
    color_labels = ["Black", "Red", "Blue", "Orange", "Green", "Yellow", "White"]

    c = canvas.Canvas(filepath, pagesize=(pw, ph))

    # ── Quadrant split (~50 rows / ~50 cols each) ─────────────────────────
    Q = 50
    def make_ranges(total):
        splits = list(range(0, total, Q)) + [total]
        return [(splits[i], splits[i + 1]) for i in range(len(splits) - 1)]

    row_ranges = make_ranges(rows)
    col_ranges = make_ranges(cols)
    total_quads = len(row_ranges) * len(col_ranges)

    # ── Helper: dice count legend table ──────────────────────────────────
    def draw_legend(lx, ly):
        """Draw legend with top-left at (lx, ly). Returns bottom y."""
        rh = 12
        cw = [52, 32, 48]
        c.setFont("Helvetica-Bold", 8)
        x = lx
        for i, hdr in enumerate(["Color", "Face", "Count"]):
            c.setFillColor(lightgrey)
            c.rect(x, ly - rh, cw[i], rh, stroke=1, fill=1)
            c.setFillColor(black)
            c.drawCentredString(x + cw[i] / 2, ly - rh + 3, hdr)
            x += cw[i]
        ly -= rh
        c.setFont("Helvetica", 8)
        for i in range(7):
            x = lx
            bg, _ = color_map[i]
            cnt = sum(row.count(i) for row in grid)
            for j, txt in enumerate([color_labels[i], f"{i} face", f"{cnt}"]):
                c.setFillColor(bg if j == 0 else white)
                c.rect(x, ly - rh, cw[j], rh, stroke=1, fill=1)
                tc = black if (i in [5, 6] and j == 0) else (white if j == 0 else black)
                c.setFillColor(tc)
                c.drawCentredString(x + cw[j] / 2, ly - rh + 3, txt)
                x += cw[j]
            ly -= rh
        return ly - 6

    # ══════════════════════════════════════════════════════════════════════
    # PAGE 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(pw / 2, ph - margin - 16,
                        f"Pipcasso Dice Map — '{project_name}'")

    info_top = ph - margin - 40
    c.setFont("Helvetica", 9)
    for i, line in enumerate([
        f"Grid: {cols} W × {rows} H  |  Total dice: {rows * cols}",
        "Instructions: Match the numbers to the dice faces shown at right.",
        "2's & 3's: arrange dots diagonally, bottom-left to top-right.",
        "6's: arrange with dots aligned vertically.",
        "0 face: colour a '1' die face with a black marker.",
    ]):
        c.drawString(margin, info_top - i * 13, line)

    draw_legend(pw * 0.72, info_top)

    # Overview grid — colour only, no numbers
    header_bottom = info_top - 5 * 13 - 10
    ov_avail_w = pw - 2 * margin
    ov_avail_h = header_bottom - margin
    ov_cell = min(ov_avail_w / cols, ov_avail_h / rows)
    ov_w = ov_cell * cols
    ov_h = ov_cell * rows
    ov_x0 = margin + (ov_avail_w - ov_w) / 2
    ov_y0 = margin + (ov_avail_h - ov_h) / 2

    for r in range(rows):
        for ci in range(cols):
            val = grid[r][ci]
            bg, _ = color_map[val]
            c.setFillColor(bg)
            c.rect(ov_x0 + ci * ov_cell,
                   ov_y0 + (rows - 1 - r) * ov_cell,
                   ov_cell, ov_cell, fill=1, stroke=0)

    c.setStrokeColor(black)
    c.setLineWidth(0.5)
    c.rect(ov_x0, ov_y0, ov_w, ov_h, fill=0, stroke=1)

    c.showPage()

    # ══════════════════════════════════════════════════════════════════════
    # PAGES 2+ — QUADRANT DETAIL PAGES
    # ══════════════════════════════════════════════════════════════════════
    quad_num = 0
    for ri, (r_start, r_end) in enumerate(row_ranges):
        for ci, (c_start, c_end) in enumerate(col_ranges):
            quad_num += 1
            q_rows = r_end - r_start
            q_cols = c_end - c_start

            # Thumbnail sits in top-right corner
            thumb_sz  = 1.4 * inch        # 100.8 pts
            thumb_x   = pw - margin - thumb_sz
            thumb_y   = ph - margin - thumb_sz

            # Header region shares the same vertical band as the thumbnail
            header_bottom = ph - margin - thumb_sz

            # Grid area — full width below header, row-label strip on left
            label_cell = 10               # pts for row/col label cells
            ga_left    = margin + label_cell
            ga_right   = pw - margin
            ga_top     = header_bottom - 4
            ga_bottom  = margin

            avail_w    = ga_right - ga_left
            avail_h    = ga_top - ga_bottom - label_cell  # reserve top strip for col headers

            cell_size  = min(avail_w / q_cols, avail_h / q_rows)
            label_font = min(label_cell * 0.42, 4.5)
            num_font   = max(3.5, min(cell_size * 0.58, 8.5))

            g_w  = cell_size * q_cols
            g_h  = cell_size * q_rows
            gx0  = ga_left + (avail_w - g_w) / 2   # centre grid horizontally
            gy0  = ga_bottom                         # grid bottom-left y

            # ── Section header ────────────────────────────────────────
            c.setFillColor(black)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, ph - margin - 14,
                         f"Section {quad_num} of {total_quads}  —  "
                         f"Rows {r_start + 1}–{r_end},  Cols {c_start + 1}–{c_end}")
            c.setFont("Helvetica", 8)
            c.drawString(margin, ph - margin - 28,
                         f"Project: {project_name}  |  Full grid: {cols} W × {rows} H  |  "
                         f"This section: {q_cols} W × {q_rows} H")

            # ── Thumbnail ─────────────────────────────────────────────
            t_cell = thumb_sz / max(rows, cols)
            t_w    = t_cell * cols
            t_h    = t_cell * rows
            tx0    = thumb_x + (thumb_sz - t_w) / 2
            ty0    = thumb_y + (thumb_sz - t_h) / 2

            for tr in range(rows):
                for tc in range(cols):
                    val = grid[tr][tc]
                    bg, _ = color_map[val]
                    c.setFillColor(bg)
                    c.rect(tx0 + tc * t_cell,
                           ty0 + (rows - 1 - tr) * t_cell,
                           t_cell, t_cell, fill=1, stroke=0)

            c.setStrokeColor(black)
            c.setLineWidth(0.5)
            c.rect(tx0, ty0, t_w, t_h, fill=0, stroke=1)

            # Highlight current quadrant on thumbnail
            hx = tx0 + c_start * t_cell
            hy = ty0 + (rows - r_end) * t_cell
            hw = q_cols * t_cell
            hh = q_rows * t_cell
            c.setFillColor(Color(1, 0.4, 0, 0.35))   # semi-transparent orange fill
            c.rect(hx, hy, hw, hh, fill=1, stroke=0)
            c.setStrokeColor(Color(0.85, 0.15, 0))
            c.setLineWidth(1.0)
            c.rect(hx, hy, hw, hh, fill=0, stroke=1)

            c.setFont("Helvetica", 5.5)
            c.setFillColor(black)
            c.drawCentredString(tx0 + t_w / 2, ty0 - 7, "Grid overview")

            # ── Column headers ────────────────────────────────────────
            col_hdr_y = gy0 + g_h     # bottom of column-header row
            c.setFont("Helvetica", label_font)
            for col_i in range(q_cols):
                actual_col = c_start + col_i
                is_tenth   = (actual_col + 1) % 10 == 0
                cx         = gx0 + col_i * cell_size
                c.setFillColor(darkgrey if is_tenth else lightgrey)
                c.setStrokeColor(black)
                c.setLineWidth(0.3)
                c.rect(cx, col_hdr_y, cell_size, label_cell, fill=1, stroke=1)
                c.setFillColor(white if is_tenth else black)
                c.drawCentredString(cx + cell_size / 2,
                                    col_hdr_y + label_cell * 0.28,
                                    f"C{actual_col + 1}")

            # ── Grid rows ─────────────────────────────────────────────
            for row_i in range(q_rows):
                actual_row = r_start + row_i
                gy         = gy0 + (q_rows - 1 - row_i) * cell_size

                # Row label
                is_tenth_r = (actual_row + 1) % 10 == 0
                c.setFillColor(darkgrey if is_tenth_r else lightgrey)
                c.setStrokeColor(black)
                c.setLineWidth(0.3)
                c.rect(gx0 - label_cell, gy, label_cell, cell_size, fill=1, stroke=1)
                c.setFillColor(white if is_tenth_r else black)
                c.setFont("Helvetica", label_font)
                c.drawCentredString(gx0 - label_cell / 2,
                                    gy + cell_size * 0.28,
                                    f"R{actual_row + 1}")

                for col_i in range(q_cols):
                    actual_col = c_start + col_i
                    val        = grid[actual_row][actual_col]
                    bg, fg     = color_map[val]
                    cx         = gx0 + col_i * cell_size

                    # Cell fill + light grid stroke
                    c.setFillColor(bg)
                    c.rect(cx, gy, cell_size, cell_size, fill=1, stroke=0)
                    c.setStrokeColor(white)
                    c.setLineWidth(0.3)
                    c.rect(cx, gy, cell_size, cell_size, fill=0, stroke=1)

                    # Dice-face number
                    c.setFillColor(fg)
                    c.setFont("Helvetica", num_font)
                    c.drawCentredString(cx + cell_size / 2,
                                        gy + cell_size * 0.28,
                                        str(val))

                    # Bold separator — right edge every 10th column
                    if (actual_col + 1) % 10 == 0:
                        c.setStrokeColor(darkgrey)
                        c.setLineWidth(1.5)
                        c.line(cx + cell_size, gy, cx + cell_size, gy + cell_size)

                    # Bold separator — top edge at quadrant start and every 10th row
                    if row_i == 0 or actual_row % 10 == 0:
                        c.setStrokeColor(darkgrey)
                        c.setLineWidth(1.5)
                        c.line(cx, gy + cell_size, cx + cell_size, gy + cell_size)

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
