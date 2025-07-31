#!/usr/bin/env python3
"""
Test script to debug why all predictions are 0
"""

import os
import json
import torch
from PIL import Image
import pandas as pd

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from enhanced_pipeline.prediction import ModelLoader

def test_single_prediction():
    """Test prediction on a single model and single image"""
    
    # Load model paths
    results_path = "data/my_results/comprehensive_pipeline_results.json"
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Get model paths
    model_paths = {}
    for col, result in data['training_results'].items():
        if result.get('success', False) and 'model_path' in result:
            model_paths[col] = result['model_path']
    
    # Test with has_faces model (usually most reliable)
    test_column = 'has_faces'
    if test_column not in model_paths:
        test_column = list(model_paths.keys())[0]
    
    print(f"\U0001f9ea TESTING SINGLE PREDICTION")
    print(f"\U0001f4dd Testing column: {test_column}")
    print(f"\U0001f4c1 Model path: {model_paths[test_column]}")
    
    # Load just this one model
    loader = ModelLoader({test_column: model_paths[test_column]}, verbose=True)
    load_results = loader.load_models()
    
    if not load_results['loaded']:
        print("\u274c Failed to load model!")
        return
    
    model = loader.get_loaded_models()[test_column]
    print(f"\u2705 Model loaded: {type(model)}")
    
    # Find a test image - look for existing extracted frames
    test_image = None
    possible_dirs = [
        'temp_frames_1753974601',  # From your last run
        'data/clip01/frames',
        'data/clip02/frames', 
        'data/clip03/frames',
        'data/clip04/frames',
        'data/clip05/frames'
    ]
    
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image = os.path.join(dir_path, file)
                    break
            if test_image:
                break
    
    if not test_image:
        print("\u274c No test image found!")
        return
    
    print(f"\U0001f5bc\ufe0f Test image: {test_image}")
    
    # Test different prediction methods
    print(f"\n\U0001f50d TESTING PREDICTION METHODS:")
    
    # Method 1: predict_image (should work)
    if hasattr(model, 'predict_image'):
        print(f"\u2705 Testing predict_image method...")
        try:
            for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                result = model.predict_image(test_image, threshold=threshold)
                print(f"   Threshold {threshold}: {result} (type: {type(result)})")
        except Exception as e:
            print(f"   \u274c predict_image failed: {e}")
    
    # Method 2: Manual prediction
    print(f"\n\U0001f527 Testing manual prediction...")
    try:
        manual_result = test_manual_prediction(model, test_image)
        print(f"   Manual result: {manual_result}")
    except Exception as e:
        print(f"   \u274c Manual prediction failed: {e}")
    
    # Method 3: Check what the CSV actually contains
    print(f"\n\U0001f4ca CHECKING ACTUAL CSV OUTPUT:")
    csv_path = "/users/9/reine097/projects/auto-label-emotions/data/clip05/out/pred.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Check a few specific values
        print(f"   CSV shape: {df.shape}")
        print(f"   First few {test_column} values: {df[test_column].head(10).tolist()}")
        
        if f'{test_column}_prob' in df.columns:
            probs = df[f'{test_column}_prob'].head(10)
            print(f"   First few {test_column}_prob values: {probs.tolist()}")
            print(f"   Probability stats: min={probs.min():.4f}, max={probs.max():.4f}, mean={probs.mean():.4f}")
            
            # Check if probabilities are all very low (below threshold)
            all_probs = df[f'{test_column}_prob']
            above_05 = (all_probs > 0.5).sum()
            above_01 = (all_probs > 0.1).sum() 
            print(f"   Frames with prob > 0.5: {above_05} / {len(all_probs)}")
            print(f"   Frames with prob > 0.1: {above_01} / {len(all_probs)}")
    
    # Method 4: Test with lower threshold
    print(f"\n\U0001f3af TESTING WITH LOWER THRESHOLDS:")
    if hasattr(model, 'predict_image'):
        try:
            for threshold in [0.01, 0.05, 0.1]:
                result = model.predict_image(test_image, threshold=threshold)
                print(f"   Very low threshold {threshold}: {result}")
        except Exception as e:
            print(f"   \u274c Low threshold test failed: {e}")

def test_manual_prediction(model, image_path):
    """Manual prediction test"""
    import torchvision.transforms as transforms
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    # Get model
    net = model.model
    net.eval()
    
    with torch.no_grad():
        outputs = net(input_tensor)
        print(f"   Raw outputs: {outputs}")
        print(f"   Output shape: {outputs.shape}")
        
        # Different activation functions
        if outputs.shape[-1] == 1:
            sigmoid_prob = float(torch.sigmoid(outputs)[0])
            raw_value = float(outputs[0])
            print(f"   Raw value: {raw_value}")
            print(f"   Sigmoid probability: {sigmoid_prob}")
        elif outputs.shape[-1] == 2:
            softmax_probs = torch.softmax(outputs, dim=-1)
            prob_pos = float(softmax_probs[0, 1])
            print(f"   Softmax probs: {softmax_probs}")  
            print(f"   Positive class prob: {prob_pos}")
        
        return {
            'raw_output': outputs.tolist(),
            'sigmoid': float(torch.sigmoid(outputs)[0]) if outputs.shape[-1] == 1 else None
        }

if __name__ == "__main__":
    test_single_prediction()