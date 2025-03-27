import sys
import torch
from torchvision import transforms

from cdni.deep_learning.pytorch_image_classifier_png_data import ImageClassifier, predict_image
from cdni.deep_learning.emotions import emotions

def predict(image_path, model_path):
    label_map = {emotions[i]: i for i in range(len(emotions))}
    num_classes = len(label_map)
    model = ImageClassifier(num_classes)
    device = 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # This scales pixels to [0.0, 1.0]
    ])
    prediction = predict_image(model, image_path, transform, label_map, device='cpu')

    return prediction


if __name__ == "__main__":
    image_path = sys.argv[1]
    model_path = 'best_model.pth'
    prediction = predict(image_path, model_path)
    print(f'prediction: {prediction}')
