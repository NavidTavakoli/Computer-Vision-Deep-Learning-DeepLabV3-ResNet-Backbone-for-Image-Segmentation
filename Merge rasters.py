import os
import rasterio
from rasterio.merge import merge
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def read_raster_data(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read only the first band
        transform = src.transform
        return data, transform

def merge_rasters_efficiently(input_dir, output_path):
    raster_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.tif')]

    # Pre-read raster data before multiprocessing
    raster_data = [read_raster_data(rf) for rf in raster_files]

    # Use ProcessPoolExecutor to parallelize processing (if necessary, for other tasks)
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Example parallel task (not necessary for reading data now)
        results = list(executor.map(some_other_processing_task, raster_data))

    # Merging without multiprocessing, since data is already in memory
    mosaic, out_trans = merge([data[0] for data in raster_data], 
                              transforms=[data[1] for data in raster_data])

    # Prepare metadata for output
    with rasterio.open(raster_files[0]) as src:
        out_meta = src.meta.copy()

    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "compress": "lzw"
    })

    # Write the output mosaic raster
    with rasterio.open(output_path, 'w', **out_meta) as dest:
        dest.write(mosaic)

    print("Mosaic created successfully.")

# Define some_other_processing_task if needed
def some_other_processing_task(data):
    # Example placeholder function
    return data

# Example usage
merge_rasters_efficiently("Test", "Test/output_file.tif")

