#!/home/navid/Thesis/.venv/bin/python3




import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from shapelysmooth import taubin_smooth

# Read the shapefile
gdf = gpd.read_file('/home/navid/Thesis/Test/mask new.shp')

# Open the GeoTIFF image file
with rasterio.open('/home/navid/Thesis/Test/sample.tif') as src:
    # Read the image
    tiff_image = src.read(1)

    # Get the extent of the image
    extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

# Set Matplotlib backend to TkAgg for interactive display
import matplotlib
matplotlib.use('TkAgg')

# Plot the GeoTIFF image
plt.imshow(tiff_image, extent=extent, cmap='gray')

# Plot the geometries from the shapefile
gdf.plot(ax=plt.gca(), edgecolor='red', facecolor='none')

# Display the plot interactively
#plt.show()

# Apply Taubin smoothing to each geometry in the GeoDataFrame
smoothed_geometries = []
for geom in gdf.geometry:
    smoothed_geom = taubin_smooth(geom, factor=0.6, mu=-0.3, steps=10)
    smoothed_geometries.append(smoothed_geom)

# Create a new GeoDataFrame with the smoothed geometries
smoothed_gdf = gpd.GeoDataFrame(geometry=smoothed_geometries, crs=gdf.crs)

# Save the smoothed mask as a new shapefile
smoothed_gdf.to_file("/home/navid/Thesis/Test/smoothed_mask.shp")
