import sys
import argparse
from PIL import Image
import rembg
import io
import os

'''
# Basic usage (same as original behavior but without deleting files)
python script.py data/tvt/pixar

# With custom output size
python script.py data/tvt/pixar --width 512 --height 512

# With separate output directory
python script.py data/tvt/pixar --output-dir processed_images

# To delete original files (use with caution!)
python script.py data/tvt/pixar --delete-originals
'''

def find_files_recursive(directory, skip_extensions=None):
    """
    Recursively find files in a directory, skipping files with certain extensions.
    
    Args:
        directory (str): Directory path to search
        skip_extensions (list): List of extensions to skip
        
    Returns:
        list: List of file paths
    """
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


def remove_bg_and_resize(input_path, output_path, size=(256, 256), delete_original=False):
    """
    Remove background from an image and resize it.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save output image
        size (tuple): Size to resize image to (width, height)
        delete_original (bool): Whether to delete the original file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if output file already exists
        if os.path.exists(output_path):
            print(f"Output file already exists: {output_path}")
            return False
        
        # Remove background - process entirely in memory
        with open(input_path, 'rb') as f:
            input_data = f.read()
            output_data = rembg.remove(input_data)
        
        # Use PIL to resize the image - process in memory
        image = Image.open(io.BytesIO(output_data)).convert('RGBA')
        image_resized = image.resize(size, Image.Resampling.LANCZOS)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the processed image
        image_resized.save(output_path, 'PNG')
        print(f"Image saved to {output_path}")
        
        # Optionally delete original
        if delete_original:
            os.remove(input_path)
            print(f"Deleted original: {input_path}")
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Remove backgrounds from images and resize them.')
    parser.add_argument('directory', type=str, help='Directory to search for images')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: same as input)')
    parser.add_argument('--width', type=int, default=256, help='Output width (default: 256)')
    parser.add_argument('--height', type=int, default=256, help='Output height (default: 256)')
    parser.add_argument('--delete-originals', action='store_true', help='Delete original files after processing')
    
    args = parser.parse_args()
    
    # Skip already processed files
    skip_extensions = ["_output.png", "_output.jpg"]
    found_files = find_files_recursive(args.directory, skip_extensions)
    
    # Track statistics
    processed_count = 0
    skipped_count = 0
    failed_count = 0
    
    for file_path in found_files:
        # Only process image files
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # Determine output path
            if args.output_dir:
                # Get relative path from input directory
                rel_path = os.path.relpath(file_path, args.directory)
                # Create output path in output directory
                output_file_path = os.path.join(args.output_dir, os.path.splitext(rel_path)[0] + '_output.png')
            else:
                # Save in same directory as input
                output_file_path = f'{os.path.splitext(file_path)[0]}_output.png'
            
            print(f"Processing: {file_path}")
            success = remove_bg_and_resize(
                file_path, 
                output_file_path, 
                size=(args.width, args.height),
                delete_original=args.delete_originals
            )
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
        else:
            print(f"Skipping non-image file: {file_path}")
            skipped_count += 1
    
    # Print summary
    print("\nSummary:")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {failed_count}")


if __name__ == '__main__':
    main()