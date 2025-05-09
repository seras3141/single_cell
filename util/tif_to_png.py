from skimage.io import imread
import os
from glob import glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm


tif_path = 'data/EXP 3 - Timlapse Brightfield/Timelapse_Plate_2172'
png_path = 'data/EXP 3 - Timlapse Brightfield/Timelapse_Plate_2172_png'

os.makedirs(png_path, exist_ok=True)

X_names = sorted(glob(os.path.join(tif_path, '*.tif')))

for f in tqdm(X_names):
    # Open the TIF file
    with Image.open(f) as img:
        # Save the image as PNG
        out_name = os.path.join(png_path, f"{Path(f).stem}.png")
        # print(img.size, out_name)
        img.save(out_name, 'PNG')


