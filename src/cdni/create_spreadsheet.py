import pandas as pd
import os
from os import listdir
from os.path import isfile, join

def get_folders_in_directory(dir_path):
    """
    Returns a list of folder names in the specified directory.
    """
    folder_list = [entry.name for entry in os.scandir(dir_path) if entry.is_dir()]
    return folder_list


df = pd.DataFrame(columns=['name', 'emotion', 'number'])

base_folder = '/users/9/reine097/projects/auto-label-emotions/data/FERG_DB_256'
character_folders = get_folders_in_directory(base_folder)
for character_folder in character_folders:
    character_emotion_folders = get_folders_in_directory(os.path.join(base_folder, character_folder))
    for character_emotion in character_emotion_folders:
        character_emotion_parts = character_emotion.split('_')
        emotion = character_emotion_parts[1]
        cartoon_file_folder = os.path.join(base_folder, character_folder, character_emotion)
        only_files = [f for f in listdir(cartoon_file_folder) if isfile(join(cartoon_file_folder, f))]
        for f in only_files:
            f_parts = f.split('_')
            number = f_parts[2][:-4]
            new_row = pd.DataFrame([{'name': character_folder, 'emotion': emotion, 'number': number}])
            df = pd.concat([df, new_row], ignore_index=True)
df.to_csv('./data/FERG_DB_256.csv', index=False)
