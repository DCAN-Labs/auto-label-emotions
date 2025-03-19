import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

df = pd.read_csv('./data/FERG_DB_256.csv')
class_names = ['anger','disgust','fear','joy','neutral',
               'sadness','surprise']
df['emotion'] = df['emotion'].astype('category')

def get_filename(row):
   name = row['name']
   emotion = row['emotion']
   number = str(row['number'])
   
   return f'/users/9/reine097/projects/auto-label-emotions/data/FERG_DB_256/{name}/{name}_{emotion}/{name}_{emotion}_{number}.png'

df['file_path'] = df.apply(get_filename, axis=1)
X = list(df['file_path'])
y = np.array(list(df['emotion']))

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test)

def move_files(group_type, group):
    base_dir = os.path.join("./data/", group_type)
    os.makedirs(base_dir, exist_ok=True)
    for path in group:
        base_name_with_extension = os.path.basename(path)
        base_name_without_extension, _ = os.path.splitext(base_name_with_extension)
        print(base_name_without_extension)
        parts = base_name_without_extension.split('_')
        emotion = parts[1]
        emotion_dir = os.path.join(base_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        shutil.move(path, emotion_dir)
        print(f"File '{path}' moved to '{emotion_dir}'")

group_types = ['train', 'val', 'test']
groups = [X_train, X_val, X_test]
for i in range(len(group_types)):
    move_files(group_types[i], groups[i])
    