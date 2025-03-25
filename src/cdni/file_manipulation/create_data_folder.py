import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

def get_files_recursive(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

def move_files_for_training(from_dir, to_dir):
    def get_emotion(row):
        file = row['file_path']
        parts = file.split('/')
        parts_of_parts = parts[-2].split('_')
        emotion = parts_of_parts[-1]
        
        return emotion
    
    files = get_files_recursive(from_dir)
    data = {'file_path': files}
    df = pd.DataFrame(data)
    df['emotion'] = df.apply(get_emotion, axis=1)
    
    X = list(df['file_path'])
    y = np.array(list(df['emotion']))
    
    X_train, X_val_test, _, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, _, _ = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test)
    
    def move_files(group_type, group, to_dir):
        base_dir = os.path.join(to_dir, group_type)
        os.makedirs(base_dir, exist_ok=True)
        
        for path in group:
            base_name_with_extension = os.path.basename(path)
            base_name_without_extension, extension = os.path.splitext(base_name_with_extension)
            print(base_name_without_extension)
            
            parts = path.split('/')
            parts_of_parts = parts[-2].split('_')
            emotion = parts_of_parts[-1]
            name = '_'.join(parts_of_parts[:-1])
            
            emotion_dir = os.path.join(base_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            file_name = f'{name}_{emotion}_{base_name_without_extension}{extension}'
            
            if os.path.isfile(path):
                print("File exists and is a file.")
                shutil.copy(path, os.path.join(emotion_dir, file_name))
            else:
                print("File does not exist or is not a file.")
            
            print(f"File '{path}' copied to '{emotion_dir}'")
    
    group_types = ['train', 'val', 'test']
    groups = [X_train, X_val, X_test]
    
    for i in range(len(group_types)):
        move_files(group_types[i], groups[i], to_dir)
            
if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python script.py from_folder to_folder")
        sys.exit(1)
        
    from_folder = sys.argv[1]
    to_folder = sys.argv[2]
    
    # Execute main function
    move_files_for_training(from_folder, to_folder)