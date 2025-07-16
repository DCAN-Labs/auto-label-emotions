# Step 1: Extract frames from multiple videos
from frame_organizer import organize_emotion_from_multiple_clips
from mp4_frame_extractor import extract_frames_from_videos
from pytorch_cartoon_face_detector import EmotionClassifier


video_list = [
    "data/clip01/in/clip1_MLP.mp4",
    "data/clip02/in/clip2_AHKJ.mp4", 
    "data/clip03/in/clip3_MLP.mp4"
]

# Extract frames (from previous multi-video extractor)
extract_frames_from_videos(
    video_list,
    output_dir="data/combined/frames",
    interval_ms=100,
    include_video_name=True
)

# Step 2: Organize frames based on annotations
clip_mapping = {
    'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv',
    'clip2_AHKJ': 'data/clip02/in/clip2_codes_AHKJ.csv',
    'clip3_MLP': 'data/clip03/in/clip3_codes_MLP.csv'
}

results = organize_emotion_from_multiple_clips(
    clip_csv_mapping=clip_mapping,
    dataset_dir='data/combined/emotion_dataset',
    output_frames_dir='data/combined/frames/',
    label_column='c_excite_face',
    positive_emotion='excited',
    negative_emotion='not_excited',
    use_move=False
)

# Step 3: Train model (from previous PyTorch classifier)
emotion_classifier = EmotionClassifier(
    positive_emotion='excited',
    negative_emotion='not_excited'
)

# Create model first 
model = emotion_classifier.create_model(pretrained=True, freeze_features=True)

train_loader, val_loader = emotion_classifier.load_dataset('data/combined/emotion_dataset')
emotion_classifier.train_model(train_loader, val_loader)
