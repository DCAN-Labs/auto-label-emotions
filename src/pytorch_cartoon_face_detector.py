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
import time
from PIL import Image
from typing import Dict, Any, Optional, List, Tuple

class CustomCNN(nn.Module):
    """Custom CNN model for binary classification"""
    
    def __init__(self, num_classes=1, dropout_rate=0.5):
        super(CustomCNN, self).__init__()
        
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

class TransferLearningModel(nn.Module):
    """Transfer learning model using pretrained backbone"""
    
    def __init__(self, backbone='mobilenet', num_classes=1, pretrained=True, freeze_features=True):
        super(TransferLearningModel, self).__init__()
        
        # Load pretrained backbone
        if backbone == 'mobilenet':
            self.backbone = models.mobilenet_v2(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            in_features = self.backbone.fc.in_features
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            in_features = self.backbone.classifier[1].in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Freeze feature extraction layers if specified
        if freeze_features:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier based on backbone type
        if backbone == 'mobilenet':
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
        elif backbone in ['resnet18', 'resnet50']:
            self.backbone.fc = nn.Sequential(
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
        elif backbone == 'efficientnet_b0':
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

class BinaryClassifier:
    def __init__(self, 
                 task_name: str = "binary_classification",
                 class_names: Optional[Dict[int, str]] = None,
                 model_type: str = 'cnn', 
                 backbone: str = 'mobilenet',
                 img_size: int = 224, 
                 device: Optional[str] = None):
        """
        Initialize the binary classifier
        
        Args:
            task_name: Name of the classification task (e.g., 'face_detection', 'emotion_classification')
            class_names: Dictionary mapping class indices to names {0: 'negative', 1: 'positive'}
            model_type: 'cnn' for custom CNN or 'transfer' for transfer learning
            backbone: Backbone model for transfer learning ('mobilenet', 'resnet18', 'resnet50', 'efficientnet_b0')
            img_size: Input image size
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.task_name = task_name
        self.class_names = class_names or {0: 'negative', 1: 'positive'}
        self.model_type = model_type
        self.backbone = backbone
        self.img_size = img_size
        # TODO Detect device at run-time
        self.device = 'cpu' # device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print(f"Initializing {self.task_name} classifier")
        print(f"Using device: {self.device}")
        print(f"Class mapping: {self.class_names}")
        
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
            self.model = CustomCNN(**kwargs)
        elif self.model_type == 'transfer':
            self.model = TransferLearningModel(backbone=self.backbone, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        return self.model
    
    def load_dataset(self, data_dir: str, batch_size: int = 32, val_split: float = 0.2):
        """Load dataset from directory structure"""
        # Create separate datasets for train and validation with different transforms
        train_dataset_full = ImageFolder(root=data_dir, transform=self.train_transform)
        val_dataset_full = ImageFolder(root=data_dir, transform=self.val_transform)
        
        # Update class names from dataset
        dataset_class_names = {i: name for i, name in enumerate(train_dataset_full.classes)}
        print(f"Dataset classes detected: {dataset_class_names}")
        
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
        
        self.dataset_class_names = train_dataset_full.classes
        print(f"Dataset structure: {self.dataset_class_names}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, epochs: int = 50, lr: float = 0.001, 
                   weight_decay: float = 1e-4, patience: int = 10, save_path: str = 'best_model.pth'):
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
        best_val_acc = 0.0
        epochs_without_improvement = 0
        total_epochs_trained = 0
        training_start_time = time.time()
        
        print(f"Starting training for {self.task_name}...")
        print(f"Target classes: {self.class_names}")
        
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
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss_avg,
                    'val_acc': val_acc,
                    'model_type': self.model_type,
                    'backbone': self.backbone,
                    'img_size': self.img_size,
                    'task_name': self.task_name,
                    'class_names': self.class_names
                }, save_path)
                print(f'  New best model saved to {save_path}')
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Early stopping triggered after {epoch+1} epochs')
                    total_epochs_trained = epoch + 1
                    break
            
            total_epochs_trained = epoch + 1
            print('-' * 50)
        
        # Calculate training time
        training_time = time.time() - training_start_time
        
        # Print comprehensive training summary
        self._print_training_summary(
            total_epochs_trained, training_time, best_val_loss, best_val_acc, 
            len(train_loader.dataset), len(val_loader.dataset), save_path
        )
        
        return self.train_history
    
    def _print_training_summary(self, total_epochs, training_time, best_val_loss, 
                               best_val_acc, train_samples, val_samples, model_path):
        """Print a comprehensive training summary"""
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = int(training_time % 60)
        
        print("\n" + "="*80)
        print(f"\U0001f3af TRAINING COMPLETE - {self.task_name.upper()}")
        print("="*80)
        
        # Model Configuration
        print("\U0001f4cb MODEL CONFIGURATION:")
        print(f"   Task: {self.task_name}")
        print(f"   Model Type: {self.model_type}")
        if self.model_type == 'transfer':
            print(f"   Backbone: {self.backbone}")
        print(f"   Image Size: {self.img_size}x{self.img_size}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.class_names}")
        
        # Training Statistics
        print(f"\n\U0001f4ca TRAINING STATISTICS:")
        print(f"   Total Epochs: {total_epochs}")
        print(f"   Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"   Training Samples: {train_samples:,}")
        print(f"   Validation Samples: {val_samples:,}")
        print(f"   Total Samples: {train_samples + val_samples:,}")
        
        # Performance Metrics
        print(f"\n\U0001f3c6 BEST PERFORMANCE:")
        print(f"   Best Validation Loss: {best_val_loss:.4f}")
        print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
        
        # Performance Assessment
        if best_val_acc >= 95:
            performance = "\U0001f31f EXCELLENT"
            emoji = "\U0001f389"
        elif best_val_acc >= 90:
            performance = "\u2705 VERY GOOD"
            emoji = "\U0001f44d"
        elif best_val_acc >= 80:
            performance = "\U0001f44c GOOD"
            emoji = "\U0001f44c"
        elif best_val_acc >= 70:
            performance = "\u26a0\ufe0f  FAIR"
            emoji = "\U0001f914"
        else:
            performance = "\u274c NEEDS IMPROVEMENT"
            emoji = "\U0001f615"
        
        print(f"\n{emoji} PERFORMANCE ASSESSMENT: {performance}")
        
        # Recommendations
        print(f"\n\U0001f4a1 RECOMMENDATIONS:")
        if best_val_acc >= 90:
            print("   \u2022 Excellent performance! Model is ready for deployment.")
            print("   \u2022 Consider testing on additional validation data.")
        elif best_val_acc >= 80:
            print("   \u2022 Good performance. Consider fine-tuning hyperparameters.")
            print("   \u2022 Try unfreezing more layers if using transfer learning.")
        elif best_val_acc >= 70:
            print("   \u2022 Fair performance. Consider:")
            print("     - Collecting more training data")
            print("     - Adjusting learning rate or batch size")
            print("     - Using data augmentation")
        else:
            print("   \u2022 Performance needs improvement. Consider:")
            print("     - Checking data quality and labels")
            print("     - Using a different model architecture")
            print("     - Collecting more diverse training data")
            print("     - Adjusting hyperparameters")
        
        # Model Information
        print(f"\n\U0001f4be MODEL SAVED:")
        print(f"   Location: {model_path}")
        print(f"   Ready for: Inference, evaluation, and deployment")
        
        # Final Statistics
        final_train_acc = self.train_history['train_acc'][-1] if self.train_history['train_acc'] else 0
        final_val_acc = self.train_history['val_acc'][-1] if self.train_history['val_acc'] else 0
        
        print(f"\n\U0001f4c8 FINAL EPOCH PERFORMANCE:")
        print(f"   Final Training Accuracy: {final_train_acc:.2f}%")
        print(f"   Final Validation Accuracy: {final_val_acc:.2f}%")
        
        # Overfitting Check
        if abs(final_train_acc - final_val_acc) > 10:
            print(f"   \u26a0\ufe0f  Warning: Large gap between train/val accuracy suggests overfitting")
        elif abs(final_train_acc - final_val_acc) < 5:
            print(f"   \u2705 Good: Small gap between train/val accuracy indicates good generalization")
        
        print("="*80)
        print("\U0001f680 Ready for evaluation and deployment!")
        print("="*80)
    
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
        
        print(f'{self.task_name} Test Results:')
        print(f'  Average Loss: {avg_loss:.4f}')
        print(f'  Accuracy: {accuracy:.2f}%')
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
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
    
    def predict_image(self, image_path: str, threshold: float = 0.5) -> Dict[str, Any]:
        """Predict class for an image"""
        if self.model is None:
            raise ValueError("Model not loaded yet.")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(image_tensor)
            probability = torch.sigmoid(output).item()
            predicted_class = int(probability > threshold)
        
        return {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': float(probability),
            'is_positive': bool(predicted_class == 1)
        }
    
    def predict_frame(self, frame, threshold: float = 0.5) -> Dict[str, Any]:
        """Predict class for a video frame"""
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
            predicted_class = int(probability > threshold)
        
        return {
            'predicted_class': predicted_class,
            'class_name': self.class_names[predicted_class],
            'confidence': float(probability),
            'is_positive': bool(predicted_class == 1)
        }
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Process a video file and classify each frame"""
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
        
        with tqdm(total=total_frames, desc=f'Processing video for {self.task_name}') as pbar:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict class for current frame
                result = self.predict_frame(frame, threshold)
                result['frame_number'] = frame_count
                frame_results.append(result)
                
                # Add text overlay to frame
                text = f"{result['class_name']}: {result['confidence']:.2f}"
                color = (0, 255, 0) if result['is_positive'] else (0, 0, 255)
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
        ax1.set_title(f'{self.task_name} - Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title(f'{self.task_name} - Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'backbone': self.backbone,
            'img_size': self.img_size,
            'task_name': self.task_name,
            'class_names': self.class_names,
            'train_history': self.train_history
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, **model_kwargs):
        """Load a saved model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Create model with saved configuration
        self.model_type = checkpoint.get('model_type', self.model_type)
        self.backbone = checkpoint.get('backbone', self.backbone)
        self.img_size = checkpoint.get('img_size', self.img_size)
        self.task_name = checkpoint.get('task_name', self.task_name)
        self.class_names = checkpoint.get('class_names', self.class_names)
        
        self.create_model(**model_kwargs)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        print(f"Model loaded from {filepath}")
        print(f"Task: {self.task_name}")
        print(f"Classes: {self.class_names}")
        return self.model

# Convenience classes for specific tasks
class FaceDetector(BinaryClassifier):
    """Face detection classifier"""
    def __init__(self, model_type='transfer', backbone='mobilenet', img_size=224, device=None):
        super().__init__(
            task_name="face_detection",
            class_names={0: 'no_face', 1: 'face'},
            model_type=model_type,
            backbone=backbone,
            img_size=img_size,
            device=device
        )
    
    def has_face(self, image_path: str, threshold: float = 0.5) -> bool:
        """Check if image contains a face"""
        result = self.predict_image(image_path, threshold)
        return result['is_positive']

class EmotionClassifier(BinaryClassifier):
    """Emotion classification (happy/sad, positive/negative, etc.)"""
    def __init__(self, positive_emotion='happy', negative_emotion='sad', 
                 model_type='transfer', backbone='mobilenet', img_size=224, device=None):
        super().__init__(
            task_name="emotion_classification",
            class_names={0: negative_emotion, 1: positive_emotion},
            model_type=model_type,
            backbone=backbone,
            img_size=img_size,
            device=device
        )
    
    def is_positive_emotion(self, image_path: str, threshold: float = 0.5) -> bool:
        """Check if image shows positive emotion"""
        result = self.predict_image(image_path, threshold)
        return result['is_positive']
    
def main():
    """Example usage for different tasks"""
    
    # Emotion Classification
    print("\n=== Emotion Classification Example ===")
    emotion_classifier = EmotionClassifier(
        positive_emotion='excited',
        negative_emotion='not_excited',
        model_type='transfer',
        backbone='resnet18'
    )
    
    # Create and train model
    model = emotion_classifier.create_model(pretrained=True, freeze_features=True)
    print(f"Emotion detection model created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Load dataset and train
    train_loader, val_loader = emotion_classifier.load_dataset('data/clip01/out/emotion_dataset', batch_size=32)
    history = emotion_classifier.train_model(train_loader, val_loader, epochs=50)

if __name__ == '__main__':
    main()
