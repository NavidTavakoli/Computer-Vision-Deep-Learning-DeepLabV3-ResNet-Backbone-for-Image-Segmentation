import subprocess

def reduce_tiff_size(input_file, output_file, compression='LZW', resize_factor=0.5):
    """
    Reduces the size of a TIFF file using compression and resizing.
    
    Args:
    input_file (str): Path to the input TIFF file.
    output_file (str): Path to save the reduced TIFF file.
    compression (str): Compression method. Default is 'LZW'.
        Possible options: 'JPEG', 'LZW', 'DEFLATE', 'NONE'.
    resize_factor (float): Resizing factor. Default is 0.5 (halves the dimensions).
    """
    try:
        # Construct the GDAL command
        gdal_command = [
            "gdal_translate",
            "-of", "GTiff",
            "-co", "COMPRESS=" + compression,
            "-outsize", f"{int(100 * resize_factor)}%", f"{int(100 * resize_factor)}%",
            input_file, output_file
        ]

        # Execute the GDAL command
        subprocess.run(gdal_command, check=True)

        print("Size reduction successful!")
    except subprocess.CalledProcessError as e:
        print("Error:", e)

# Example usage:
input_file = 'Test/1.tif'
output_file = 'Test/1_compressed.tif'
compression_method = 'LZW'  # LZW is chosen for lossless compression
resize_factor = 0.5  # Reduce the image size to 50%
reduce_tiff_size(input_file, output_file, compression_method, resize_factor)
