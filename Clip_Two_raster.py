from osgeo import gdal

# Define the paths to the input files and the output file
image_path = 'Test/output_RGB.tif'
mask_path = 'Test/Full Mask Turin.tif'
clipped_output_path = 'Test/Clipped_Image.tif'

# Open the mask dataset to determine its properties
mask_ds = gdal.Open(mask_path, gdal.GA_ReadOnly)
if mask_ds is None:
    raise FileNotFoundError("Mask file could not be opened.")

# Get the geotransform and use it to calculate the bounding coordinates
mask_geo_transform = mask_ds.GetGeoTransform()
min_x = mask_geo_transform[0]
max_y = mask_geo_transform[3]
max_x = min_x + mask_geo_transform[1] * mask_ds.RasterXSize
min_y = max_y + mask_geo_transform[5] * mask_ds.RasterYSize

# Close the mask dataset
mask_ds = None

# Use gdal.Warp to clip the image using the mask's extent and set pixel size
warp_options = gdal.WarpOptions(
    format='GTiff',
    outputBounds=[min_x, min_y, max_x, max_y],  # Specify the coordinates of the bounding box
    dstSRS='EPSG:3003',  # Assuming the CRS should be preserved, adjust if different
    xRes=0.4,  # Set the pixel size in X direction
    yRes=0.4,  # Set the pixel size in Y direction
    cropToCutline=True,
    resampleAlg=gdal.GRA_Cubic  # Use a high-quality resampling algorithm
)

# Perform the warp operation to clip the image
image_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
if image_ds is None:
    raise FileNotFoundError("Image file could not be opened.")

result_ds = gdal.Warp(destNameOrDestDS=clipped_output_path, srcDSOrSrcDSTab=image_ds, options=warp_options)
if result_ds is None:
    raise Exception("Error during the clipping operation.")

# Close the datasets
result_ds = None
image_ds = None

print(f"Image has been successfully clipped and saved to {clipped_output_path}")
