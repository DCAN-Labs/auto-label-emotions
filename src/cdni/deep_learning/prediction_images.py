import os
from os import listdir
from os.path import isfile, join
from matplotlib import transforms
from torch.utils.data import Dataset, DataLoader

from cdni.deep_learning.predict_image import predict
from cdni.deep_learning.pytorch_image_classifier_png_data import ImageClassifier, PNGImageDataset, evaluate_model

if __name__ == 'main':
    base_dir = 'data/tvt/pixar'
    divisions = ['train', 'val', 'test']
    model_path = '/home/feczk001/shared/data/auto_label_emotions/models/ferg00.pth'
    num_classes = len(emotions)
    # For RGBA images, we need to handle 4 channels
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # This scales pixels to [0.0, 1.0]
    ])
    model = ImageClassifier(num_classes)
    print("Loading testing dataset...")
    test_dataset = PNGImageDataset(
        image_dir=f'{base_dir}',
        transform=transform,
        label_map=train_dataset.label_map
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    evaluate_model(model, test_loader, device='cuda')
