import os
from os import listdir
from os.path import isfile, join
import shutil

from cdni.deep_learning.emotions import emotions

input_dir = 'data/tvt/pixar'
output_dir = 'data/png_dataset/pixar'
for group in ['train', 'val', 'test']:
    group_folder = os.path.join(input_dir, group)
    for emotion in emotions:
        src_emotion_folder = os.path.join(group_folder, emotion)
        only_files = [f for f in listdir(src_emotion_folder) if isfile(join(src_emotion_folder, f))]
        dest_emotion_path = os.path.join(output_dir, emotion)
        os.makedirs(dest_emotion_path, exist_ok=True)
        for f in only_files:
            source_path = join(src_emotion_folder, f)
            destination_path = join(dest_emotion_path, f)

            try:
                shutil.copy(source_path, destination_path)
                print(f"File '{source_path}' copied successfully to '{destination_path}'")
            except FileNotFoundError:
                print(f"Error: Source file '{source_path}' not found.")
            except PermissionError:
                print(f"Error: Permission denied to access '{source_path}' or '{destination_path}'.")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
