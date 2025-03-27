import os
from os import listdir
from os.path import isfile, join
import shutil
import argparse
from tqdm import tqdm  # For progress bars

# Verify the import or provide a fallback
try:
    from cdni.deep_learning.emotions import emotions
    print(f"Found {len(emotions)} emotions: {emotions}")
except ImportError:
    print("Warning: Could not import emotions list. Using default empty list.")
    emotions = []

def copy_emotion_dataset(input_dir, output_dir, emotions):
    """Copy emotion dataset files maintaining emotion categories but flattening train/val/test."""
    
    # Count total files for progress tracking
    total_files = 0
    for group in ['train', 'val', 'test']:
        group_folder = os.path.join(input_dir, group)
        if not os.path.exists(group_folder):
            print(f"Warning: Group folder {group_folder} does not exist.")
            continue
            
        for emotion in emotions:
            src_emotion_folder = os.path.join(group_folder, emotion)
            if not os.path.exists(src_emotion_folder):
                print(f"Warning: Emotion folder {src_emotion_folder} does not exist.")
                continue
                
            only_files = [f for f in listdir(src_emotion_folder) if isfile(join(src_emotion_folder, f))]
            total_files += len(only_files)
    
    # Process the files with progress bar
    with tqdm(total=total_files, desc="Copying files") as pbar:
        # Track duplicate filenames
        filename_counts = {}
        
        for group in ['train', 'val', 'test']:
            group_folder = os.path.join(input_dir, group)
            if not os.path.exists(group_folder):
                continue
                
            for emotion in emotions:
                src_emotion_folder = os.path.join(group_folder, emotion)
                if not os.path.exists(src_emotion_folder):
                    continue
                    
                only_files = [f for f in listdir(src_emotion_folder) if isfile(join(src_emotion_folder, f))]
                dest_emotion_path = os.path.join(output_dir, emotion)
                os.makedirs(dest_emotion_path, exist_ok=True)
                
                for f in only_files:
                    source_path = join(src_emotion_folder, f)
                    
                    # Handle potential duplicates
                    if f in filename_counts:
                        filename_counts[f] += 1
                        base, ext = os.path.splitext(f)
                        dest_filename = f"{base}_{group}_{filename_counts[f]}{ext}"
                    else:
                        filename_counts[f] = 1
                        dest_filename = f
                        
                    destination_path = join(dest_emotion_path, dest_filename)
                    
                    try:
                        shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
                        pbar.update(1)
                    except FileNotFoundError:
                        print(f"\nError: Source file '{source_path}' not found.")
                    except PermissionError:
                        print(f"\nError: Permission denied to access '{source_path}' or '{destination_path}'.")
                    except Exception as e:
                        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Copy emotion dataset with restructuring.')
    parser.add_argument('--input', default='data/tvt/pixar', help='Input directory with train/val/test structure')
    parser.add_argument('--output', default='data/png_dataset/pixar', help='Output directory for restructured data')
    args = parser.parse_args()
    
    copy_emotion_dataset(args.input, args.output, emotions)
    print("Dataset copying completed!")