import rasterio
from rasterio.windows import Window
import os

def clip_images(folder_path, output_folder, new_width, new_height):
    for filename in os.listdir(folder_path):
        if filename.endswith(".tif"):
            img_path = os.path.join(folder_path, filename)
            
            # Ensure the output directory exists
            final_output_folder = os.path.join(output_folder)
            os.makedirs(final_output_folder, exist_ok=True)
            
            with rasterio.open(img_path) as src:
                # Calculate the position of the new window
                old_width, old_height = src.width, src.height
                x_off = (old_width - new_width) // 2
                y_off = (old_height - new_height) // 2
                
                # Create a window to read
                window = Window(x_off, y_off, new_width, new_height)
                kwargs = src.meta.copy()
                kwargs.update({
                    'height': new_height,
                    'width': new_width,
                    'transform': rasterio.windows.transform(window, src.transform)})

                # Read the data from the window and write it to a new file
                output_file_path = os.path.join(final_output_folder, filename)
                with rasterio.open(output_file_path, 'w', **kwargs) as dst:
                    for i in range(1, src.count + 1):  # Loop through bands
                        dst.write(src.read(i, window=window), i)
            print(f"{filename} has been processed and saved.")

# Specify your folder paths and desired dimensions here
folder_path = "Test"
output_folder = "Test/Output"
new_width = 20224
new_height = 20224

clip_images(folder_path, output_folder, new_width, new_height)
