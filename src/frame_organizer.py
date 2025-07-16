import pandas as pd
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

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
               timestamp_column: str,
               label_column: str,
               class_names: Dict[int, str],
               dataset_dir: str,
               output_frames_dir: str,
               file_extension: str = "jpg",
               use_move: bool = True,
               verbose: bool = True) -> bool:
    """
    Move or copy a frame to the appropriate dataset directory based on classification label
    
    Args:
        row: DataFrame row containing timestamp and label columns
        timestamp_column: Name of the timestamp column (e.g., 'onset_milliseconds')
        label_column: Name of the binary label column (e.g., 'has_faces', 'is_happy')
        class_names: Dictionary mapping label values to directory names {0: 'negative', 1: 'positive'}
        dataset_dir: Base dataset directory
        output_frames_dir: Source directory with extracted frames
        file_extension: File extension without dot (default: 'jpg')
        use_move: If True, move files; if False, copy files
        verbose: If True, print operations
    
    Returns:
        bool: True if successful, False otherwise
    """
    timestamp_ms = row[timestamp_column]
    label_value = row[label_column]
    
    # Determine target directory based on label
    if label_value in class_names:
        category = class_names[label_value]
        output_dir = os.path.join(dataset_dir, category)
    else:
        if verbose:
            print(f"Warning: Unknown label value '{label_value}' for {label_column}. Skipping.")
        return False
    
    # Ensure destination directory exists
    ensure_directory_exists(output_dir)
    
    # Generate file paths
    source_path = generate_filename_from_timestamp(timestamp_ms, file_extension, output_frames_dir)
    destination_path = generate_filename_from_timestamp(timestamp_ms, file_extension, output_dir)
    
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
                           timestamp_column: str,
                           label_column: str,
                           class_names: Dict[int, str],
                           dataset_dir: str,
                           output_frames_dir: str,
                           file_extension: str = "jpg",
                           use_move: bool = True,
                           verbose: bool = True) -> Dict[str, Any]:
    """
    Organize frames from CSV annotations into dataset structure
    
    Args:
        csv_file: Path to CSV file with annotations
        timestamp_column: Name of the timestamp column (e.g., 'onset_milliseconds')
        label_column: Name of the binary label column (e.g., 'has_faces', 'is_happy')
        class_names: Dictionary mapping label values to directory names {0: 'negative', 1: 'positive'}
        dataset_dir: Base dataset directory
        output_frames_dir: Source directory with extracted frames
        file_extension: File extension without dot (default: 'jpg')
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
    required_columns = [timestamp_column, label_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return {'error': f'Missing columns: {missing_columns}'}
    
    # Initialize statistics
    stats = {
        'total_rows': len(df),
        'successful_operations': 0,
        'failed_operations': 0,
        'class_distribution': {class_name: 0 for class_name in class_names.values()},
        'skipped_files': 0
    }
    
    # Create base directories
    for class_name in class_names.values():
        ensure_directory_exists(os.path.join(dataset_dir, class_name))
    
    if verbose:
        print(f"Processing {len(df)} rows from '{csv_file}'...")
        print(f"Timestamp column: {timestamp_column}")
        print(f"Label column: {label_column}")
        print(f"Class mapping: {class_names}")
        print(f"Source directory: {output_frames_dir}")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Operation: {'Move' if use_move else 'Copy'}")
        print("-" * 50)
    
    # Process each row
    for index, row in df.iterrows():
        success = move_frame(
            row, timestamp_column, label_column, class_names,
            dataset_dir, output_frames_dir, file_extension, use_move, verbose
        )
        
        if success:
            stats['successful_operations'] += 1
            label_value = row[label_column]
            if label_value in class_names:
                class_name = class_names[label_value]
                stats['class_distribution'][class_name] += 1
        else:
            stats['failed_operations'] += 1
    
    # Print final statistics
    if verbose:
        print("-" * 50)
        print("Operation Summary:")
        print(f"  Total rows processed: {stats['total_rows']}")
        print(f"  Successful operations: {stats['successful_operations']}")
        print(f"  Failed operations: {stats['failed_operations']}")
        print(f"  Class distribution:")
        for class_name, count in stats['class_distribution'].items():
            print(f"    {class_name}: {count}")
        print(f"  Success rate: {(stats['successful_operations']/stats['total_rows']*100):.1f}%")
    
    return stats

def validate_dataset_structure(dataset_dir: str, class_names: Dict[int, str], 
                             file_extension: str = "jpg") -> Dict[str, Any]:
    """
    Validate the created dataset structure
    
    Args:
        dataset_dir: Base dataset directory
        class_names: Dictionary mapping label values to directory names
        file_extension: File extension to count (default: 'jpg')
    
    Returns:
        dict: Validation results
    """
    results = {
        'dataset_dir_exists': os.path.exists(dataset_dir),
        'class_directories': {},
        'total_files': 0
    }
    
    for label_value, class_name in class_names.items():
        class_dir = os.path.join(dataset_dir, class_name)
        class_info = {
            'exists': os.path.exists(class_dir),
            'count': 0
        }
        
        if class_info['exists']:
            files = [f for f in os.listdir(class_dir) if f.endswith(f'.{file_extension}')]
            class_info['count'] = len(files)
        
        results['class_directories'][class_name] = class_info
        results['total_files'] += class_info['count']
    
    print("Dataset Validation Results:")
    print(f"  Dataset directory exists: {results['dataset_dir_exists']}")
    for class_name, info in results['class_directories'].items():
        print(f"  {class_name} directory exists: {info['exists']}")
        print(f"  {class_name} files: {info['count']}")
    print(f"  Total files: {results['total_files']}")
    
    return results

# Convenience functions for common use cases
def organize_face_detection_frames(csv_file: str,
                                 dataset_dir: str,
                                 output_frames_dir: str,
                                 timestamp_column: str = 'onset_milliseconds',
                                 label_column: str = 'has_faces',
                                 use_move: bool = True,
                                 verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function for face detection frame organization
    """
    class_names = {0: 'no_face', 1: 'face'}
    return organize_frames_from_csv(
        csv_file, timestamp_column, label_column, class_names,
        dataset_dir, output_frames_dir, "jpg", use_move, verbose
    )

def organize_emotion_frames(csv_file: str,
                          dataset_dir: str,
                          output_frames_dir: str,
                          timestamp_column: str = 'onset_milliseconds',
                          label_column: str = 'is_happy',
                          use_move: bool = True,
                          verbose: bool = True) -> Dict[str, Any]:
    """
    Convenience function for emotion detection frame organization
    """
    class_names = {0: 'not_happy', 1: 'happy'}
    return organize_frames_from_csv(
        csv_file, timestamp_column, label_column, class_names,
        dataset_dir, output_frames_dir, "jpg", use_move, verbose
    )

def organize_binary_classification_frames(csv_file: str,
                                        dataset_dir: str,
                                        output_frames_dir: str,
                                        timestamp_column: str,
                                        label_column: str,
                                        negative_class_name: str = 'negative',
                                        positive_class_name: str = 'positive',
                                        use_move: bool = True,
                                        verbose: bool = True) -> Dict[str, Any]:
    """
    Generic convenience function for any binary classification
    """
    class_names = {0: negative_class_name, 1: positive_class_name}
    return organize_frames_from_csv(
        csv_file, timestamp_column, label_column, class_names,
        dataset_dir, output_frames_dir, "jpg", use_move, verbose
    )

if __name__ == '__main__':
    # Custom emotion detection
    print("\n=== Emotion Detection ===")
    emotion_stats = organize_binary_classification_frames(
        csv_file='data/clip01/in/clip1_codes_MLP.csv',
        dataset_dir='data/clip01/out/emotion_dataset',
        output_frames_dir='data/clip01/out/output_frames/',
        timestamp_column='onset_milliseconds',
        label_column='c_excite_face',  
        negative_class_name='not_excited',
        positive_class_name='excited',
        use_move=False,
        verbose=True
    )
