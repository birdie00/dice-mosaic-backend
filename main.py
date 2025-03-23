from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import os
from uuid import uuid4
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# 🔗 Your Render backend URL (no trailing slash)
RENDER_BASE_URL = "https://dice-mosaic-backend.onrender.com"

# Create static folder for hosting files
os.makedirs("static", exist_ok=True)

# ✅ Create FastAPI app
app = FastAPI()

# ✅ CORS Middleware (after app is defined)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev — allows localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static PDFs
app.mount("/static", StaticFiles(directory="static"), name="static")

# 🔧 Generate dice grid
def generate_dice_grid(image, grid_size):
    image = image.resize((grid_size, grid_size)).convert("L")  # Grayscale
    pixels = list(image.getdata())
    values = [min(6, max(0, pixel // 40)) for pixel in pixels]
    return [values[i:i + grid_size] for i in range(0, len(values), grid_size)]

# 📄 Generate dice map PDF
def generate_dice_map_pdf(grid, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    cell_size = 12
    margin = 40
    y_offset = height - margin

    c.setFont("Courier", 8)
    c.drawString(margin, y_offset, "Dice Map")
    y_offset -= 20

    for row in grid:
        x = margin
        for val in row:
            c.drawString(x, y_offset, str(val))
            x += cell_size
        y_offset -= cell_size

        if y_offset < margin:
            c.showPage()
            y_offset = height - margin

    c.save()

# 🔍 /analyze endpoint
@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    grid_size: int = Form(100)
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Generate the grid
    grid = generate_dice_grid(image, grid_size)

    # Save PDF
    filename = f"dice_map_{uuid4().hex}.pdf"
    pdf_path = f"static/{filename}"
    generate_dice_map_pdf(grid, pdf_path)

    # Build public link
    dice_map_url = f"{RENDER_BASE_URL}/static/{filename}"

    return {
        "grid": grid,
        "dice_map_url": dice_map_url
    }
