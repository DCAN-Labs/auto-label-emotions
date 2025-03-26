import sys
from PIL import Image
import rembg
import os

def main(directory_to_search):
    def find_files_recursive_os_walk(directory, skip_extensions=None):
        if skip_extensions is None:
            skip_extensions = []
        
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                # Skip files with extensions in skip_extensions
                if not any(file.endswith(ext) for ext in skip_extensions):
                    file_path = os.path.join(root, file)
                    file_paths.append(file_path)
        return file_paths
    
    def remove_bg_and_resize(input_path, output_path):
        try:
            # Remove background
            with open(input_path, 'rb') as f:
                input_data = f.read()
                output_data = rembg.remove(input_data)
            
            # Save the transparent image to a temporary file
            temp_path = 'temp_rgba.png'
            try:
                with open(temp_path, 'wb') as f:
                    f.write(output_data)
                
                # Open with PIL for resizing
                image = Image.open(temp_path).convert('RGBA')
                image_resized = image.resize((256, 256), Image.LANCZOS)
                image_resized.save(output_path, 'PNG')
                print(f"Image saved to {output_path}")
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                os.remove(input_path)
                    
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    
    # Skip already processed files
    skip_extensions = ["_output.png", "_output.jpg"]
    found_files = find_files_recursive_os_walk(directory_to_search, skip_extensions)
    
    for file_path in found_files:
        # Only process image files
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            output_file_path = f'{file_path[:-4]}_output.png'  # Save as PNG to preserve transparency
            print(f"Processing: {file_path}")
            remove_bg_and_resize(file_path, output_file_path)
        else:
            print(f"Skipping non-image file: {file_path}")

if __name__ == '__main__':
    directory_to_search = "data/tvt/pixar"
    main(directory_to_search)