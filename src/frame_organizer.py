import pandas as pd
import os
import shutil
from pathlib import Path
from typing import Optional

def generate_filename_from_timestamp(timestamp_ms: int, extension: str, base_dir: str) -> str:
    """
    Generate filename from timestamp matching the frame extractor format
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        extension: File extension (without dot)
        base_dir: Base directory path
    
    Returns:
        Full file path
    """
    timestamp_str = f"{timestamp_ms:08.0f}ms"
    filename = f"frame_{timestamp_str}.{extension.lower()}"
    return os.path.join(base_dir, filename)

def ensure_directory_exists(directory: str) -> None:
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def move_frame(row: pd.Series, 
               dataset_dir: str = 'data/clip01/out/dataset',
               output_frames_dir: str = 'data/clip01/out/output_frames/',
               use_move: bool = True,
               verbose: bool = True) -> bool:
    """
    Move or copy a frame to the appropriate dataset directory based on face detection
    
    Args:
        row: DataFrame row containing 'onset_milliseconds' and 'has_faces'
        dataset_dir: Base dataset directory
        output_frames_dir: Source directory with extracted frames
        use_move: If True, move files; if False, copy files
        verbose: If True, print operations
    
    Returns:
        bool: True if successful, False otherwise
    """
    onset_milliseconds = row['onset_milliseconds']
    has_faces = row['has_faces']
    
    # Fix the logic: has_faces == 1 means there ARE faces
    if has_faces == 1 or has_faces > 0:
        output_dir = os.path.join(dataset_dir, 'face')
        category = 'face'
    else:
        output_dir = os.path.join(dataset_dir, 'no_face')
        category = 'no_face'
    
    # Ensure destination directory exists
    ensure_directory_exists(output_dir)
    
    # Generate file paths
    source_path = generate_filename_from_timestamp(onset_milliseconds, "jpg", output_frames_dir)
    destination_path = generate_filename_from_timestamp(onset_milliseconds, "jpg", output_dir)
    
    try:
        # Check if source file exists
        if not os.path.exists(source_path):
            if verbose:
                print(f"Warning: Source file '{source_path}' not found.")
            return False
        
        # Check if destination already exists
        if os.path.exists(destination_path):
            if verbose:
                print(f"Warning: Destination file '{destination_path}' already exists. Skipping.")
            return False
        
        # Move or copy the file
        if use_move:
            shutil.move(source_path, destination_path)
            operation = "moved"
        else:
            shutil.copy2(source_path, destination_path)  # copy2 preserves metadata
            operation = "copied"
        
        if verbose:
            print(f"File '{os.path.basename(source_path)}' {operation} to '{category}' directory.")
        
        return True
        
    except FileNotFoundError:
        if verbose:
            print(f"Error: Source file '{source_path}' not found.")
        return False
    except PermissionError:
        if verbose:
            print(f"Error: Permission denied when accessing '{source_path}' or '{destination_path}'.")
        return False
    except Exception as e:
        if verbose:
            print(f"An error occurred with file '{source_path}': {e}")
        return False

def organize_frames_from_csv(csv_file: str, 
                           dataset_dir: str = 'data/clip01/out/dataset',
                           output_frames_dir: str = 'data/clip01/out/output_frames/',
                           use_move: bool = True,
                           verbose: bool = True) -> dict:
    """
    Organize frames from CSV annotations into dataset structure
    
    Args:
        csv_file: Path to CSV file with annotations
        dataset_dir: Base dataset directory
        output_frames_dir: Source directory with extracted frames
        use_move: If True, move files; if False, copy files
        verbose: If True, print operations
    
    Returns:
        dict: Statistics about the operation
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        return {'error': 'CSV file not found'}
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {'error': f'Error reading CSV: {e}'}
    
    # Validate required columns
    required_columns = ['onset_milliseconds', 'has_faces']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return {'error': f'Missing columns: {missing_columns}'}
    
    # Initialize statistics
    stats = {
        'total_rows': len(df),
        'successful_operations': 0,
        'failed_operations': 0,
        'face_frames': 0,
        'no_face_frames': 0,
        'skipped_files': 0
    }
    
    # Create base directories
    ensure_directory_exists(os.path.join(dataset_dir, 'face'))
    ensure_directory_exists(os.path.join(dataset_dir, 'no_face'))
    
    if verbose:
        print(f"Processing {len(df)} rows from '{csv_file}'...")
        print(f"Source directory: {output_frames_dir}")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Operation: {'Move' if use_move else 'Copy'}")
        print("-" * 50)
    
    # Process each row
    for index, row in df.iterrows():
        success = move_frame(row, dataset_dir, output_frames_dir, use_move, verbose)
        
        if success:
            stats['successful_operations'] += 1
            if row['has_faces'] == 1 or row['has_faces'] > 0:
                stats['face_frames'] += 1
            else:
                stats['no_face_frames'] += 1
        else:
            stats['failed_operations'] += 1
    
    # Print final statistics
    if verbose:
        print("-" * 50)
        print("Operation Summary:")
        print(f"  Total rows processed: {stats['total_rows']}")
        print(f"  Successful operations: {stats['successful_operations']}")
        print(f"  Failed operations: {stats['failed_operations']}")
        print(f"  Frames with faces: {stats['face_frames']}")
        print(f"  Frames without faces: {stats['no_face_frames']}")
        print(f"  Success rate: {(stats['successful_operations']/stats['total_rows']*100):.1f}%")
    
    return stats

def validate_dataset_structure(dataset_dir: str = 'data/clip01/out/dataset') -> dict:
    """
    Validate the created dataset structure
    
    Args:
        dataset_dir: Base dataset directory
    
    Returns:
        dict: Validation results
    """
    face_dir = os.path.join(dataset_dir, 'face')
    no_face_dir = os.path.join(dataset_dir, 'no_face')
    
    results = {
        'face_dir_exists': os.path.exists(face_dir),
        'no_face_dir_exists': os.path.exists(no_face_dir),
        'face_count': 0,
        'no_face_count': 0,
        'total_files': 0
    }
    
    if results['face_dir_exists']:
        face_files = [f for f in os.listdir(face_dir) if f.endswith('.jpg')]
        results['face_count'] = len(face_files)
    
    if results['no_face_dir_exists']:
        no_face_files = [f for f in os.listdir(no_face_dir) if f.endswith('.jpg')]
        results['no_face_count'] = len(no_face_files)
    
    results['total_files'] = results['face_count'] + results['no_face_count']
    
    print("Dataset Validation Results:")
    print(f"  Face directory exists: {results['face_dir_exists']}")
    print(f"  No-face directory exists: {results['no_face_dir_exists']}")
    print(f"  Files with faces: {results['face_count']}")
    print(f"  Files without faces: {results['no_face_count']}")
    print(f"  Total files: {results['total_files']}")
    
    return results

if __name__ == '__main__':
    # Configuration
    csv_file = 'data/clip01/in/clip1_codes_MLP.csv'
    dataset_dir = 'data/clip01/out/dataset'
    output_frames_dir = 'data/clip01/out/output_frames/'
    
    # Organize frames (use move=False to copy instead of move)
    stats = organize_frames_from_csv(
        csv_file=csv_file,
        dataset_dir=dataset_dir,
        output_frames_dir=output_frames_dir,
        use_move=False,  # Set to False to copy instead of move
        verbose=True
    )
    
    # Validate the result
    if 'error' not in stats:
        print("\n" + "="*50)
        validate_dataset_structure(dataset_dir)
