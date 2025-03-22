
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), grid_size: int = 40):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")  # Convert to grayscale
    width, height = image.size

    tile_w, tile_h = width // grid_size, height // grid_size
    dice_grid = []

    for y in range(grid_size):
        row = []
        for x in range(grid_size):
            box = (x * tile_w, y * tile_h, (x + 1) * tile_w, (y + 1) * tile_h)
            tile = image.crop(box)
            avg_brightness = int(sum(tile.getdata()) / (tile_w * tile_h))

            # Map brightness: 0 (darkest, blank dice) → 6 (brightest, full dots)
            if avg_brightness < 30:
                dice_val = 0
            elif avg_brightness < 60:
                dice_val = 1
            elif avg_brightness < 100:
                dice_val = 2
            elif avg_brightness < 140:
                dice_val = 3
            elif avg_brightness < 180:
                dice_val = 4
            elif avg_brightness < 220:
                dice_val = 5
            else:
                dice_val = 6

            row.append(dice_val)
        dice_grid.append(row)

    return JSONResponse(content={"grid": dice_grid})
