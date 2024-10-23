import os
from osgeo import gdal
from tqdm import tqdm

def set_band_color_interpretation(raster_path):
    dataset = gdal.Open(raster_path, gdal.GA_Update)  # Open the raster for updating
    if not dataset:
        print("Error opening the raster file.")
        return

    try:
        # Set the color interpretation
        band1 = dataset.GetRasterBand(1)
        band2 = dataset.GetRasterBand(2)
        band3 = dataset.GetRasterBand(3)
        
        band1.SetColorInterpretation(gdal.GCI_RedBand)   # Red channel
        band2.SetColorInterpretation(gdal.GCI_GreenBand) # Green channel
        band3.SetColorInterpretation(gdal.GCI_BlueBand)  # Blue channel
        
        print("Color interpretation set to RGB for the three bands.")
    except RuntimeError as e:
        print(f"An error occurred: {e}")
    finally:
        dataset = None  # Close the dataset

def calculate_windows(width, height, tile_width, tile_height, overlap):
    """Calculate window positions and sizes for tiling."""
    stride_x = int(tile_width * (1 - overlap))
    stride_y = int(tile_height * (1 - overlap))
    for y in range(0, height, stride_y):
        for x in range(0, width, stride_x):
            yield (x, y, tile_width, tile_height)

def split_raster_gdal(input_path, output_dir, tile_size, overlap_percentage):
    """Split a raster file into tiles using GDAL."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = gdal.Open(input_path)
    width, height = dataset.RasterXSize, dataset.RasterYSize
    tile_width, tile_height = tile_size

    overlap = overlap_percentage / 100.0
    windows = list(calculate_windows(width, height, tile_width, tile_height, overlap))

    # tqdm loop to show progress
    for i, (x_offset, y_offset, win_width, win_height) in enumerate(tqdm(windows, desc="Processing tiles", unit="tile"), start=1):
        if x_offset + win_width > width:
            win_width = width - x_offset
        if y_offset + win_height > height:
            win_height = height - y_offset

        out_path = os.path.join(output_dir, f"tile_{i:04}.tif")
        srcwin = (x_offset, y_offset, win_width, win_height)
        dst_ds = gdal.Translate(out_path, dataset, format='GTiff', srcWin=srcwin)
        dst_ds = None  # Close and save file

# Example usage
if __name__ == "__main__":
    raster_path = "Test/Test1.tif"
    output_dir = "train/images"
    tile_size = (256, 256)
    overlap_percentage = 0

    set_band_color_interpretation(raster_path)
    split_raster_gdal(raster_path, output_dir, tile_size, overlap_percentage)
