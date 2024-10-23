import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping, shape
from rasterio.features import shapes

# Open the raster file
with rasterio.open('Test/population raster level 1.tif') as src:
    image = src.read()  # Read the raster into an array
    transform = src.transform

# Generate masks for each unique color and vectorize
def mask_to_polygons(mask):
    all_polygons = []
    for geom, val in shapes(mask.astype(np.int16), mask=(mask == 1), transform=transform):
        all_polygons.append(shape(geom))
    return all_polygons

# Assuming the raster is RGB and we're interpreting colors as unique combinations
unique_colors = np.unique(image.reshape(-1, image.shape[0]), axis=0)
color_polygons = []

for color in unique_colors:
    # Create a mask for each unique color
    mask = np.all(image == color[:, None, None], axis=0)
    # Convert the mask to polygons
    polygons = mask_to_polygons(mask)
    color_polygons.append({'polygons': polygons, 'color': color})

# Create a GeoDataFrame
gdf = gpd.GeoDataFrame(columns=['geometry', 'R', 'G', 'B'])

for item in color_polygons:
    for poly in item['polygons']:
        gdf = gdf.append({'geometry': poly, 'R': item['color'][0], 'G': item['color'][1], 'B': item['color'][2]}, ignore_index=True)

# Save to file
gdf.to_file('Test/output_vector.geojson', driver='GeoJSON')
