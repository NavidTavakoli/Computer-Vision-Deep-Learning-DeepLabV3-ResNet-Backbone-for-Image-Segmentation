from osgeo import gdal

def apply_geotransform_and_projection(source_tif, target_tif):
    # Open the source TIFF to read its geotransform and projection
    src_ds = gdal.Open(source_tif)
    if src_ds is None:
        print(f"Failed to open source file: {source_tif}")
        return
    
    # Read the geotransform and projection from the source
    src_geotransform = src_ds.GetGeoTransform()
    src_projection = src_ds.GetProjection()
    
    # Open the target TIFF in update mode
    tgt_ds = gdal.Open(target_tif, gdal.GA_Update)
    if tgt_ds is None:
        print(f"Failed to open target file: {target_tif}")
        return
    
    # Apply the geotransform and projection to the target
    tgt_ds.SetGeoTransform(src_geotransform)
    tgt_ds.SetProjection(src_projection)
    
    # Close the datasets
    src_ds = None
    tgt_ds = None
    print(f"Updated {target_tif} with geotransform and projection from {source_tif}.")

# Replace these with your actual file paths
source_tif = 'Test/Full Mask Turin.tif'
target_tif = 'Test/Full Mask Turin gray.tif'
apply_geotransform_and_projection(source_tif, target_tif)
