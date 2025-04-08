from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, landscape, portrait
from reportlab.lib.colors import black, white, gray, red, lightgrey
import os


def draw_section_preview(c, full_width, full_height, view_x, view_y, view_w, view_h, x_offset, y_offset):
    preview_w = 80
    preview_h = 80 * full_height / full_width
    cell_size_w = preview_w / full_width
    cell_size_h = preview_h / full_height
    top_left_x = x_offset
    top_left_y = y_offset - preview_h

    c.setStrokeColor(lightgrey)
    c.setFillColor(lightgrey)
    c.rect(top_left_x, top_left_y, preview_w, preview_h, fill=1)

    c.setStrokeColor(red)
    highlight_x = top_left_x + view_x * cell_size_w
    highlight_y = top_left_y + (full_height - view_y - view_h) * cell_size_h
    c.rect(highlight_x, highlight_y, view_w * cell_size_w, view_h * cell_size_h, fill=0, stroke=1)


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
            c.setFont("Helvetica", number_font_size + 1.5)
            c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - ((number_font_size + 1.5) / 2) * 0.3, str(val))

    for x in range(width):
        label = f"C{start_x + x + 1}"
        px = grid_left + x * cell_size
        py = grid_top + cell_size
        c.setFillColor(white)
        c.setStrokeColor(gray if ghost and x == width - 1 else black)
        c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)
        c.setFillColor(gray if ghost and x == width - 1 else black)
        c.setFont("Helvetica", label_font_size)
        c.drawCentredString(px + cell_size / 2, py - cell_size / 2 - (label_font_size / 2) * 0.3, label)

    for y in range(height):
        label = f"R{start_y + y + 1}"
        px = grid_left - cell_size
        py = grid_top - y * cell_size
        c.setFillColor(white)
        c.setStrokeColor(gray if ghost and y == height - 1 else black)
        c.rect(px, py - cell_size, cell_size, cell_size, fill=1, stroke=1)
        c.setFillColor(gray if ghost and y == height - 1 else black)
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
        3: (0, 128, 0, white),
        4: (255, 165, 0, black),
        5: (255, 255, 0, black),
        6: (255, 255, 255, black),
    }

    # Page 1: Full Map Overview
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

    top_of_grid = page_height - margin - 80 - (len(instructions) * 16) - 20
    available_height = top_of_grid - margin
    available_width = page_width - 2 * margin
    cell_size = min(available_width / width, available_height / height)

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
            page_width - margin - 90,
            page_height - margin - 20
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
