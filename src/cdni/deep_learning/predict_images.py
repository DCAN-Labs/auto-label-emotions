import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from cdni.deep_learning.pytorch_image_classifier_png_data import ImageClassifier, PNGImageDataset, evaluate_model
from cdni.deep_learning.emotions import emotions

if __name__ == '__main__':
    base_dir = 'data/png_dataset/pixar'
    model_path = '/home/feczk001/shared/data/auto_label_emotions/models/ferg00.pth'
    device = 'cpu'
    
    # Define model with the same number of classes as during training
    num_classes = len(emotions)
    model = ImageClassifier(num_classes)
    
    # Load pre-trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    
    # For RGBA images, we need to handle 4 channels
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # This scales pixels to [0.0, 1.0]
    ])
    
    # Create label map from emotions list
    label_map = {emotion: i for i, emotion in enumerate(emotions)}
    
    print("Loading testing dataset...")
    test_dataset = PNGImageDataset(
        image_dir=f'{base_dir}',
        transform=transform,
        label_map=label_map  # Use the same label mapping as during training
    )
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    accuracy, cm = evaluate_model(model, test_loader, device=device)
    print(f'accuracy: {accuracy}')
    print(f'cm: {cm}')
    