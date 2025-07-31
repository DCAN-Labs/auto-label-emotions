#!/usr/bin/env python3
"""
Quick diagnostic script to identify the prediction issue
Run this first to understand what's happening
"""

import os
import json
import torch
import pandas as pd
from PIL import Image, ImageDraw

# Force CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def quick_diagnosis():
    """Quick diagnosis of the prediction issue"""
    
    print("ð QUICK DIAGNOSIS - PREDICTION ISSUE")
    print("="*50)
    
    # Step 1: Check if model files exist
    results_path = "data/my_results/comprehensive_pipeline_results.json"
    if not os.path.exists(results_path):
        print("â Step 1 FAILED: No results file found")
        return
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    model_paths = {}
    for col, result in data['training_results'].items():
        if result.get('success', False) and 'model_path' in result:
            model_paths[col] = result['model_path']
    
    print(f"â Step 1 PASSED: Found {len(model_paths)} model paths")
    
    for col, path in model_paths.items():
        exists = os.path.exists(path)
        print(f"   {col}: {'â' if exists else 'â'} {path}")
    
    # Step 2: Check existing CSV output
    csv_path = "data/clip05/out/pred.csv"
    if os.path.exists(csv_path):
        print(f"\nâ Step 2: Analyzing existing CSV")
        df = pd.read_csv(csv_path)
        
        pred_cols = [col for col in df.columns if not col.endswith('_prob') 
                    and col not in ['onset_milliseconds', 'frame_path']]
        
        print(f"   CSV shape: {df.shape}")
        print(f"   Prediction columns: {pred_cols}")
        
        for col in pred_cols:
            unique_vals = df[col].unique()
            print(f"   {col}: unique values = {unique_vals}")
            
            if f'{col}_prob' in df.columns:
                prob_stats = df[f'{col}_prob'].describe()
                print(f"   {col}_prob stats: min={prob_stats['min']:.6f}, max={prob_stats['max']:.6f}, mean={prob_stats['mean']:.6f}")
    else:
        print(f"\nâ Step 2 SKIPPED: No existing CSV found at {csv_path}")
    
    # Step 3: Test model loading
    print(f"\nð Step 3: Testing model loading")
    
    try:
        from enhanced_pipeline.prediction import ModelLoader
        
        # Test with just one model
        test_col = list(model_paths.keys())[0]
        test_path = model_paths[test_col]
        
        print(f"   Testing {test_col} model...")
        
        loader = ModelLoader({test_col: test_path}, verbose=False)
        results = loader.load_models()
        
        if results['loaded']:
            print(f"   â Model loaded successfully")
            
            # Test prediction
            model = loader.get_loaded_models()[test_col]
            
            # Create a simple test image
            test_img = create_simple_test_image()
            
            # Test different prediction methods
            test_prediction_methods(model, test_img, test_col)
            
        else:
            print(f"   â Model loading failed")
            
    except Exception as e:
        print(f"   â Step 3 FAILED: {e}")
        import traceback
        traceback.print_exc()

def create_simple_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (224, 224), color='lightblue')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple face
    draw.ellipse([50, 50, 174, 174], fill='peachpuff', outline='black', width=2)
    draw.ellipse([70, 80, 90, 100], fill='white', outline='black')
    draw.ellipse([134, 80, 154, 100], fill='white', outline='black')
    draw.ellipse([75, 85, 85, 95], fill='black')
    draw.ellipse([139, 85, 149, 95], fill='black')
    draw.arc([85, 125, 139, 155], start=0, end=180, fill='red', width=3)
    
    test_path = "quick_test_image.jpg"
    img.save(test_path)
    return test_path

def test_prediction_methods(model, image_path, column):
    """Test different prediction methods on the model"""
    print(f"   ð§ª Testing prediction methods for {column}:")
    
    # Method 1: predict_image
    if hasattr(model, 'predict_image'):
        try:
            result = model.predict_image(image_path, threshold=0.5)
            print(f"      predict_image: {result} (type: {type(result)})")
        except Exception as e:
            print(f"      predict_image failed: {e}")
    
    # Method 2: Manual prediction
    try:
        result = manual_test_prediction(model, image_path)
        print(f"      manual_prediction: {result}")
    except Exception as e:
        print(f"      manual_prediction failed: {e}")

def manual_test_prediction(model, image_path):
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
        
        # Handle different output formats
        if outputs.shape[-1] == 1:
            raw_val = float(outputs[0, 0])
            sigmoid_prob = float(torch.sigmoid(outputs)[0, 0])
            return f"raw={raw_val:.4f}, sigmoid={sigmoid_prob:.4f}"
        elif outputs.shape[-1] == 2:
            softmax_probs = torch.softmax(outputs, dim=1)
            prob_pos = float(softmax_probs[0, 1])
            return f"softmax_pos={prob_pos:.4f}"
        else:
            return f"unexpected_shape={outputs.shape}"

if __name__ == "__main__":
    quick_diagnosis()
    
    # Clean up
    if os.path.exists("quick_test_image.jpg"):
        os.remove("quick_test_image.jpg")