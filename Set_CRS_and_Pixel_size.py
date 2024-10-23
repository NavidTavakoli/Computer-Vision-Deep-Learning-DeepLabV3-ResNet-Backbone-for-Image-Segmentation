from osgeo import gdal

# Define the input and output file paths
input_tif_path = 'Test/Mask.tif'
output_tif_path = 'Test/path_to_mask.tif'

# Open the input raster
src_ds = gdal.Open(input_tif_path, gdal.GA_ReadOnly)
if src_ds is None:
    raise FileNotFoundError("The specified file could not be opened.")

# Define the target CRS and pixel size
target_crs = 'EPSG:3003'
pixel_size_x = 0.4
pixel_size_y = 0.4

# Set up the options for warping (including reprojection and resampling)
warp_options = gdal.WarpOptions(
    format='GTiff',
    xRes=pixel_size_x,
    yRes=pixel_size_y,
    dstSRS=target_crs,
    resampleAlg=gdal.GRA_NearestNeighbour  # You can choose another resampling algorithm as needed
)

# Perform the warping (reprojection and resizing)
result_ds = gdal.Warp(destNameOrDestDS=output_tif_path, srcDSOrSrcDSTab=src_ds, options=warp_options)
if result_ds is None:
    raise Exception("Error during warping operation.")

# Close the datasets
result_ds = None
src_ds = None

print(f"File has been reprojected and resized, saved to {output_tif_path}")
