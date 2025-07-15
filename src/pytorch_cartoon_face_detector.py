import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import json
from PIL import Image

class CartoonFaceCNN(nn.Module):
    """Custom CNN model for cartoon face detection"""
    
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(CartoonFaceCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Fourth convolutional block
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CartoonFaceMobileNet(nn.Module):
    """Transfer learning model using MobileNetV2"""
    
    def __init__(self, num_classes=1, pretrained=True, freeze_features=True):
        super(CartoonFaceMobileNet, self).__init__()
        
        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(pretrained=pretrained)
        
        # Freeze feature extraction layers if specified
        if freeze_features:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Replace classifier
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class VideoFrameDataset(Dataset):
    """Custom dataset for video frames"""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

class CartoonFaceDetector:
    def __init__(self, model_type='cnn', img_size=224, device=None):
        """
        Initialize the cartoon face detector
        
        Args:
            model_type (str): 'cnn' for custom CNN or 'mobilenet' for transfer learning
            img_size (int): Input image size
            device (str): Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_type = model_type
        self.img_size = img_size
        self.device = 'cpu' # device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"Using device: {self.device}")
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def create_model(self, **kwargs):
        """Create the model based on model_type"""
        if self.model_type == 'cnn':
            self.model = CartoonFaceCNN(**kwargs)
        elif self.model_type == 'mobilenet':
            self.model = CartoonFaceMobileNet(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        return self.model
    
    def load_dataset(self, data_dir, batch_size=32, val_split=0.2):
        """Load dataset from directory structure"""
        # Create separate datasets for train and validation with different transforms
        train_dataset_full = ImageFolder(root=data_dir, transform=self.train_transform)
        val_dataset_full = ImageFolder(root=data_dir, transform=self.val_transform)
        
        # Get indices for splitting
        dataset_size = len(train_dataset_full)
        val_size = int(dataset_size * val_split)
        train_size = dataset_size - val_size
        
        # Generate consistent indices
        indices = list(range(dataset_size))
        torch.manual_seed(42)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset_full, train_indices)
        val_dataset = Subset(val_dataset_full, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True if self.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True if self.device == 'cuda' else False
        )
        
        self.class_names = train_dataset_full.classes
        print(f"Classes: {self.class_names}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, epochs=50, lr=0.001, 
                   weight_decay=1e-4, patience=10, save_path='best_model.pth'):
        """Train the model"""
        if self.model is None:
            raise ValueError("Model not created yet. Call create_model() first.")
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                target = target.float().unsqueeze(1)  # Convert to float and add dimension for BCE loss
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = torch.sigmoid(output) > 0.5
                train_correct += (predicted == target).sum().item()
                train_total += target.size(0)
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for data, target in val_pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    target = target.float().unsqueeze(1)  # Convert to float and add dimension for BCE loss
                    
                    output = self.model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    predicted = torch.sigmoid(output) > 0.5
                    val_correct += (predicted == target).sum().item()
                    val_total += target.size(0)
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            # Calculate epoch metrics
            train_loss_avg = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Store history
            self.train_history['train_loss'].append(train_loss_avg)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss_avg)
            self.train_history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Learning rate scheduling
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss_avg)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f'  Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}')
            
            # Early stopping and model saving
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc,
                    'model_type': self.model_type,
                    'img_size': self.img_size
                }, save_path)
                print(f'  New best model saved to {save_path}')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    break
            
            print('-' * 50)
        
        return self.train_history
    
    def evaluate_model(self, test_loader):
        """Evaluate the model on test dataset"""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc='Evaluating'):
                data, target = data.to(self.device), target.to(self.device)
                target = target.float().unsqueeze(1)  # Convert to float and add dimension for BCE loss
                
                output = self.model(data)
                test_loss += criterion(output, target).item()
                
                predicted = torch.sigmoid(output) > 0.5
                correct += (predicted == target).sum().item()
                total += target.size(0)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        
        print(f'Test Results:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Accuracy: {accuracy:.2f}%')
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(all_targets, all_predictions)
        recall = recall_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions)
        
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1 Score: {f1:.4f}')
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def predict_image(self, image_path, threshold=0.5):
        """Predict if an image contains a face"""
        if self.model is None:
            raise ValueError("Model not loaded yet.")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.sigmoid(output).item()
            has_face = probability > threshold
        
        return {
            'has_face': bool(has_face),
            'confidence': float(probability)
        }
    
    def predict_frame(self, frame, threshold=0.5):
        """Predict if a video frame contains a face"""
        if self.model is None:
            raise ValueError("Model not loaded yet.")
        
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.sigmoid(output).item()
            has_face = probability > threshold
        
        return {
            'has_face': bool(has_face),
            'confidence': float(probability)
        }
    
    def process_video(self, video_path, output_path=None, threshold=0.5):
        """Process a video file and detect faces in each frame"""
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_results = []
        
        with tqdm(total=total_frames, desc='Processing video') as pbar:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict face in current frame
                result = self.predict_frame(frame, threshold)
                result['frame_number'] = frame_count
                frame_results.append(result)
                
                # Add text overlay to frame
                text = f"Face: {'YES' if result['has_face'] else 'NO'} ({result['confidence']:.2f})"
                color = (0, 255, 0) if result['has_face'] else (0, 0, 255)
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Write frame to output video
                if output_path:
                    out.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        
        return frame_results
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.train_history['train_loss']:
            raise ValueError("No training history available.")
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # Plot loss
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'img_size': self.img_size,
            'train_history': self.train_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath, **model_kwargs):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Create model with saved configuration
        self.model_type = checkpoint.get('model_type', self.model_type)
        self.img_size = checkpoint.get('img_size', self.img_size)
        
        self.create_model(**model_kwargs)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        print(f"Model loaded from {filepath}")
        return self.model

def main():
    """Example usage"""
    # Initialize detector
    detector = CartoonFaceDetector(model_type='mobilenet', img_size=224)
    
    # Create model
    model = detector.create_model(pretrained=True, freeze_features=True)
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Load dataset (uncomment when you have data)
    train_loader, val_loader = detector.load_dataset('data/clip01/out/dataset', batch_size=32)
    
    # Train model (uncomment when you have data)
    history = detector.train_model(train_loader, val_loader, epochs=50, lr=0.001)
    
    # Plot training history
    detector.plot_training_history()
    
    # Evaluate model
    metrics = detector.evaluate_model(val_loader)
    
    # Save model
    detector.save_model('cartoon_face_detector_pytorch.pth')
    
    # Load model
    detector.load_model('cartoon_face_detector_pytorch.pth')
    
    # Predict on single image
    # result = detector.predict_image('path/to/image.jpg')
    # print(f"Has face: {result['has_face']}, Confidence: {result['confidence']:.2f}")
    
    # Process video
    # frame_results = detector.process_video('input_video.mp4', 'output_video.mp4')
    
    print("PyTorch cartoon face detection model created successfully!")

if __name__ == "__main__":
    main()
