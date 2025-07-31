#!/usr/bin/env python3
"""
Quick fix for prediction module - let's inspect the actual methods available
"""

import os
import json
import torch
from PIL import Image
import torchvision.transforms as transforms

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def inspect_model_methods():
    """Inspect what methods are actually available in the trained models"""
    
    # Load a trained model to see its methods
    results_path = "data/my_results/comprehensive_pipeline_results.json"
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Get first successful model
    for col, result in data['training_results'].items():
        if result.get('success', False) and 'model_path' in result:
            model_path = result['model_path']
            print(f"\U0001f50d Inspecting model: {col}")
            print(f"\U0001f4c1 Path: {model_path}")
            
            try:
                # Try to recreate the classifier
                from pytorch_cartoon_face_detector import BinaryClassifier, FaceDetector
                
                if col == 'has_faces':
                    classifier = FaceDetector(
                        model_type='transfer',
                        backbone='mobilenet',
                        img_size=224,
                        device='cpu'
                    )
                else:
                    class_names = {0: f'not_{col}', 1: col}
                    classifier = BinaryClassifier(
                        task_name=f"{col}_classification",
                        class_names=class_names,
                        model_type='transfer',
                        backbone='mobilenet',
                        img_size=224,
                        device='cpu'
                    )
                
                # Create model and load weights
                classifier.create_model(pretrained=False, freeze_features=True)
                classifier.load_model(model_path)
                
                print(f"\u2705 Model loaded successfully")
                print(f"\U0001f527 Model type: {type(classifier)}")
                
                # List ALL methods (not just callable ones)
                all_attrs = dir(classifier)
                methods = [attr for attr in all_attrs if callable(getattr(classifier, attr, None))]
                
                print(f"\U0001f4cb All methods ({len(methods)}):")
                for method in sorted(methods):
                    if not method.startswith('_'):
                        print(f"   \u2022 {method}")
                
                # Check specific attributes
                print(f"\n\U0001f3d7\ufe0f Model structure:")
                if hasattr(classifier, 'model'):
                    print(f"   \u2022 Has 'model' attribute: {type(classifier.model)}")
                if hasattr(classifier, 'network'):  
                    print(f"   \u2022 Has 'network' attribute: {type(classifier.network)}")
                if hasattr(classifier, 'device'):
                    print(f"   \u2022 Device: {classifier.device}")
                if hasattr(classifier, 'img_size'):
                    print(f"   \u2022 Image size: {classifier.img_size}")
                
                # Try manual prediction with a dummy image
                print(f"\n\U0001f9ea Testing manual prediction:")
                test_manual_prediction(classifier)
                
                break
                
            except Exception as e:
                print(f"\u274c Error: {e}")
                continue

def test_manual_prediction(classifier):
    """Test manual prediction approach"""
    
    # Create a dummy image
    dummy_image = Image.new('RGB', (224, 224), color='red')
    dummy_path = 'temp_test_image.jpg'
    dummy_image.save(dummy_path)
    
    try:
        # Manual prediction
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(dummy_image).unsqueeze(0)
        print(f"   \U0001f4ca Input tensor shape: {input_tensor.shape}")
        
        # Get the actual model
        if hasattr(classifier, 'model'):
            net = classifier.model
            print(f"   \U0001f3af Using classifier.model: {type(net)}")
        else:
            print(f"   \u274c No 'model' attribute found")
            return
        
        net.eval()
        
        with torch.no_grad():
            outputs = net(input_tensor)
            print(f"   \U0001f4c8 Raw outputs: {outputs}")
            print(f"   \U0001f4cf Output shape: {outputs.shape}")
            
            # Try sigmoid
            if outputs.shape[-1] == 1:
                sigmoid_out = torch.sigmoid(outputs)
                prob = float(sigmoid_out[0])
                pred = 1 if prob > 0.5 else 0
                print(f"   \U0001f3af Sigmoid: prob={prob:.4f}, pred={pred}")
            
            # Try softmax  
            elif outputs.shape[-1] == 2:
                softmax_out = torch.softmax(outputs, dim=-1)
                prob = float(softmax_out[0, 1])
                pred = 1 if prob > 0.5 else 0
                print(f"   \U0001f3af Softmax: prob={prob:.4f}, pred={pred}")
            
            else:
                print(f"   \u2753 Unexpected output shape")
    
    except Exception as e:
        print(f"   \u274c Manual prediction failed: {e}")
    
    finally:
        # Clean up
        if os.path.exists(dummy_path):
            os.remove(dummy_path)

if __name__ == "__main__":
    inspect_model_methods()