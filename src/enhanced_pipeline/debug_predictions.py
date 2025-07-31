#!/usr/bin/env python3
"""
Debug script to diagnose prediction issues
"""

import os
import json
import pandas as pd
import torch
from pathlib import Path

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from enhanced_pipeline.prediction import ModelLoader, VideoPredictor

def debug_single_model_prediction():
    """Debug a single model prediction step by step"""
    
    # Load the pipeline results to get model paths
    results_path = "data/my_results/comprehensive_pipeline_results.json"
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    # Get model paths from training results
    model_paths = {}
    for col, result in data['training_results'].items():
        if result.get('success', False) and 'model_path' in result:
            model_paths[col] = result['model_path']
    
    print(f"\U0001f50d Found {len(model_paths)} trained models")
    
    # Pick one model to debug (let's use has_faces since it's usually reliable)
    debug_column = 'has_faces'
    if debug_column not in model_paths:
        debug_column = list(model_paths.keys())[0]  # Use first available
    
    model_path = model_paths[debug_column]
    print(f"\U0001f3af Debugging model: {debug_column}")
    print(f"\U0001f4c1 Model path: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"\u274c Model file not found!")
        return
    
    # Load the model
    print(f"\U0001f4e6 Loading model...")
    loader = ModelLoader({debug_column: model_path}, verbose=True)
    load_results = loader.load_models()
    
    if not load_results['loaded']:
        print(f"\u274c Failed to load model!")
        return
    
    model = loader.get_loaded_models()[debug_column]
    print(f"\u2705 Model loaded successfully")
    print(f"\U0001f527 Model type: {type(model)}")
    
    # List available methods
    methods = [m for m in dir(model) if not m.startswith('_') and callable(getattr(model, m))]
    print(f"\U0001f6e0\ufe0f Available methods: {methods[:15]}...")  # Show first 15
    
    # Try to find a test image
    test_image = None
    
    # Look for existing frames first
    temp_dirs = [d for d in os.listdir('.') if d.startswith('temp_frames_')]
    if temp_dirs:
        temp_dir = temp_dirs[0]
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_image = os.path.join(root, file)
                    break
            if test_image:
                break
    
    # If no temp frames, look in data directories
    if not test_image:
        data_dirs = ['data/clip01', 'data/clip02', 'data/clip03', 'data/clip04', 'data/clip05']
        for data_dir in data_dirs:
            frames_dir = Path(data_dir) / 'frames'
            if frames_dir.exists():
                for img_file in frames_dir.glob('*.jpg'):
                    test_image = str(img_file)
                    break
                if test_image:
                    break
    
    if not test_image:
        print("\u274c No test image found!")
        return
    
    print(f"\U0001f5bc\ufe0f Test image: {test_image}")
    
    # Try different prediction approaches
    print(f"\n\U0001f9ea TESTING PREDICTION METHODS:")
    
    # Method 1: Check for standard methods
    prediction_methods = ['predict_single_image', 'predict_image', 'predict', 'classify_image']
    
    for method_name in prediction_methods:
        if hasattr(model, method_name):
            print(f"\n\u2705 Found method: {method_name}")
            try:
                method = getattr(model, method_name)
                result = method(test_image, threshold=0.5)
                print(f"   Result: {result} (type: {type(result)})")
            except Exception as e:
                print(f"   \u274c Error calling {method_name}: {e}")
        else:
            print(f"\u274c Method not found: {method_name}")
    
    # Method 2: Manual prediction
    print(f"\n\U0001f527 TRYING MANUAL PREDICTION:")
    try:
        result = manual_predict_debug(model, test_image, debug_column)
        print(f"   Manual result: {result}")
    except Exception as e:
        print(f"   \u274c Manual prediction failed: {e}")
    
    # Method 3: Inspect model structure
    print(f"\n\U0001f3d7\ufe0f MODEL STRUCTURE:")
    if hasattr(model, 'model'):
        print(f"   Has 'model' attribute: {type(model.model)}")
    if hasattr(model, 'network'):
        print(f"   Has 'network' attribute: {type(model.network)}")
    if hasattr(model, 'classifier'):
        print(f"   Has 'classifier' attribute: {type(model.classifier)}")
        
    # Method 4: Check state dict
    print(f"\n\U0001f4be MODEL STATE:")
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        print(f"   State dict keys: {list(state_dict.keys())[:10]}...")
        
        # Check if it's a full model or just weights
        if 'model_state_dict' in state_dict:
            print("   Format: Full checkpoint with model_state_dict")
            actual_weights = state_dict['model_state_dict']
        else:
            print("   Format: Direct state dict")
            actual_weights = state_dict
            
        weight_keys = list(actual_weights.keys())[:10]
        print(f"   Weight keys: {weight_keys}...")
        
    except Exception as e:
        print(f"   \u274c Error loading state dict: {e}")

def manual_predict_debug(model, image_path, column):
    """Manual prediction with detailed debugging"""
    from PIL import Image
    import torchvision.transforms as transforms
    
    print(f"   \U0001f4f8 Loading image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    print(f"   \U0001f4cf Image size: {image.size}")
    
    # Get expected input size
    img_size = getattr(model, 'img_size', 224)
    print(f"   \U0001f3af Target size: {img_size}")
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0)
    print(f"   \U0001f504 Input tensor shape: {input_tensor.shape}")
    
    # Find the actual model
    net = None
    if hasattr(model, 'model'):
        net = model.model
        print(f"   \U0001f3af Using model.model: {type(net)}")
    elif hasattr(model, 'network'):
        net = model.network  
        print(f"   \U0001f3af Using model.network: {type(net)}")
    elif hasattr(model, 'classifier'):
        net = model.classifier
        print(f"   \U0001f3af Using model.classifier: {type(net)}")
    else:
        net = model
        print(f"   \U0001f3af Using model directly: {type(net)}")
    
    # Set eval mode
    net.eval()
    
    # Run inference
    with torch.no_grad():
        print(f"   \U0001f680 Running inference...")
        outputs = net(input_tensor)
        print(f"   \U0001f4ca Raw output shape: {outputs.shape}")
        print(f"   \U0001f4ca Raw output values: {outputs}")
        
        # Try different activation functions
        if outputs.shape[-1] == 1:
            # Single output
            sigmoid_out = torch.sigmoid(outputs)
            print(f"   \U0001f4c8 Sigmoid output: {sigmoid_out}")
            
            probability = float(sigmoid_out[0])
            prediction = 1 if probability > 0.5 else 0
            
        elif outputs.shape[-1] == 2:
            # Two outputs
            softmax_out = torch.softmax(outputs, dim=-1)
            print(f"   \U0001f4c8 Softmax output: {softmax_out}")
            
            probability = float(softmax_out[0, 1])  # Probability of positive class
            prediction = 1 if probability > 0.5 else 0
        else:
            print(f"   \u2753 Unexpected output shape: {outputs.shape}")
            probability = 0.0
            prediction = 0
        
        print(f"   \U0001f3af Final prediction: {prediction} (prob: {probability:.4f})")
        return {'prediction': prediction, 'probability': probability}

def check_csv_stats():
    """Check statistics of the generated CSV"""
    csv_path = "/users/9/reine097/projects/auto-label-emotions/data/clip05/out/pred.csv"
    
    if not os.path.exists(csv_path):
        print(f"\u274c CSV file not found: {csv_path}")
        return
    
    print(f"\U0001f4ca ANALYZING GENERATED CSV:")
    df = pd.read_csv(csv_path)
    
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {len(df.columns)}")
    
    # Check prediction columns (non-probability columns)
    pred_columns = [col for col in df.columns 
                   if col not in ['onset_milliseconds', 'frame_path'] 
                   and not col.endswith('_prob')]
    
    print(f"   Prediction columns: {len(pred_columns)}")
    
    # Check if all predictions are 0
    all_zeros = True
    for col in pred_columns:
        col_sum = df[col].sum()
        col_max = df[col].max()
        print(f"   {col}: sum={col_sum}, max={col_max}")
        if col_sum > 0:
            all_zeros = False
    
    if all_zeros:
        print(f"   \u274c ALL PREDICTIONS ARE ZERO!")
    else:
        print(f"   \u2705 Some predictions are non-zero")
    
    # Check probability columns
    prob_columns = [col for col in df.columns if col.endswith('_prob')]
    print(f"\n\U0001f4c8 PROBABILITY ANALYSIS:")
    
    for col in prob_columns[:5]:  # Check first 5 probability columns
        prob_stats = df[col].describe()
        print(f"   {col}: mean={prob_stats['mean']:.4f}, max={prob_stats['max']:.4f}")

if __name__ == "__main__":
    print("\U0001f50d PREDICTION DEBUGGING SCRIPT")
    print("="*50)
    
    # First check the CSV stats
    check_csv_stats()
    
    print("\n" + "="*50)
    
    # Then debug the model prediction
    debug_single_model_prediction()