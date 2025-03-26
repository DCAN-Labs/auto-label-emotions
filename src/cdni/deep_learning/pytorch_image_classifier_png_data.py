import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the CNN architecture
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        
        # Input shape: (batch_size, 4, 256, 256) - RGBA images
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # After 4 pooling operations, the spatial dimensions are reduced by 2^4 = 16
        # So 256/16 = 16, resulting in a 256 x 16 x 16 tensor
        self.fc1 = nn.Linear(256 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        # Convolutional layers with ReLU activation, batch normalization and pooling
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(F.relu(self.batch_norm4(self.conv4(x))))
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 256 * 16 * 16)
        
        # Fully connected layers with ReLU activation and dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Custom dataset for loading PNG images
class PNGImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, label_map=None):
        """
        Args:
            image_dir (str): Directory containing subdirectories of images, where each subdirectory name is a class
            transform (callable, optional): Optional transform to be applied on a sample
            label_map (dict, optional): Optional mapping from directory names to class indices
        """
        self.image_dir = image_dir
        self.transform = transform
        
        # If no label map is provided, create one based on subdirectory names
        if label_map is None:
            self.classes = sorted([d for d in os.listdir(image_dir) 
                               if os.path.isdir(os.path.join(image_dir, d))])
            self.label_map = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.label_map = label_map
            self.classes = sorted(label_map.keys())
        
        self.images = []
        self.labels = []
        self.valid_images = []
        
        # Collect all images and their corresponding labels
        for cls_name in self.classes:
            cls_dir = os.path.join(image_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
                
            cls_idx = self.label_map[cls_name]
            
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith('.png'):
                    img_path = os.path.join(cls_dir, img_name)
                    
                    # Pre-validate images to avoid errors during loading
                    try:
                        # Try to open the image to check if it's valid
                        with Image.open(img_path) as img:
                            # Just accessing a property to ensure image can be processed
                            _ = img.mode
                        # If successful, add to valid images
                        self.valid_images.append(img_path)
                        self.labels.append(cls_idx)
                    except Exception as e:
                        print(f"Warning: Skipping corrupted or invalid image: {img_path}. Error: {e}")
                        continue
        
        self.images = self.valid_images
        print(f"Loaded {len(self.images)} valid images out of {len(self.images)} total files with .png extension")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Open image with PIL (handles 8-bit/color RGBA PNG)
            image = Image.open(img_path).convert('RGBA')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            # This shouldn't happen since we pre-validated images, but just in case
            print(f"Error loading image at runtime: {img_path}. Error: {e}")
            # Return a placeholder image
            placeholder = torch.zeros((4, 256, 256))
            return placeholder, label

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Train the model and validate after each epoch
    """
    model.to(device)
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved with validation accuracy: {val_acc:.4f}')
    
    return model, history

# Function to evaluate the model
def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on the test set
    """
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return accuracy, cm

# Example usage
def main(data_root_folder):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define transforms for the training, validation, and testing sets
    # For RGBA images, we need to handle 4 channels
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # This scales pixels to [0.0, 1.0]
    ])
    
    # Verify data directories exist
    for data_dir in [f'{data_root_folder}/train', f'{data_root_folder}/val', f'{data_root_folder}/test']:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            print(f"Created directory: {data_dir}")
    
    # Create datasets with error handling
    try:
        print("Loading training dataset...")
        train_dataset = PNGImageDataset(
            image_dir=f'{data_root_folder}/train',
            transform=transform
        )
        
        print("Loading validation dataset...")
        val_dataset = PNGImageDataset(
            image_dir=f'{data_root_folder}/val',
            transform=transform,
            label_map=train_dataset.label_map
        )
        
        print("Loading testing dataset...")
        test_dataset = PNGImageDataset(
            image_dir=f'{data_root_folder}/test',
            transform=transform,
            label_map=train_dataset.label_map
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Print class information
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create model
    num_classes = len(train_dataset.classes)
    model = ImageClassifier(num_classes)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=20,
        device=device
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate model
    accuracy, cm = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# Example of how to predict a single image
def predict_image(model, image_path, transform, label_map, device='cuda'):
    """
    Predict the class of a single image
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGBA')
        image = transform(image).unsqueeze(0).to(device)
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            predicted_idx = preds.item()
        
        # Map predicted index to class name
        idx_to_class = {v: k for k, v in label_map.items()}
        predicted_class = idx_to_class[predicted_idx]
        
        # Get softmax probabilities
        probabilities = F.softmax(outputs, dim=1)[0]
        
        return predicted_class, probabilities.cpu().numpy()
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        return None, None

if __name__ == "__main__":
    data_root_folder = sys.argv[1]
    main(data_root_folder)
