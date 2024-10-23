import os
import rasterio
from rasterio.windows import Window
from tqdm import tqdm 

def calculate_windows(width, height, tile_width, tile_height, overlap):
    """Calculate window positions and sizes for tiling."""
    stride_x = int(tile_width * (1 - overlap))
    stride_y = int(tile_height * (1 - overlap))

    for y in range(0, height, stride_y):
        for x in range(0, width, stride_x):
            yield Window(x, y, tile_width, tile_height)

def split_raster(input_path, output_dir, tile_size, overlap_percentage):
    """Split a raster file into tiles, ensuring the output directory exists, with a progress bar."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open(input_path) as src:
        width, height = src.width, src.height
        tile_width, tile_height = tile_size

        overlap = overlap_percentage / 100.0
        windows = list(calculate_windows(width, height, tile_width, tile_height, overlap))  # Convert generator to list to know the total count

        for i, window in enumerate(tqdm(windows, desc="Splitting raster"), start=1):  # tqdm loop
            transform = src.window_transform(window)

            out_path = os.path.join(output_dir, f"tile_{i:04}.tif")

            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=window.height,
                width=window.width,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=transform,
            ) as dst:
                dst.write(src.read(window=window))

# Example usage
tile_size = (256, 256)  # Tile size in pixels (width, height)
overlap_percentage = 0  # Overlap percentage

# Splitting the main raster and its mask
#split_raster("python_files/Viena_masks.tif", "train/masks", tile_size, overlap_percentage)
split_raster("Test/Full Mask Turin.tif", "train/masks", tile_size, overlap_percentage)



