import pandas as pd
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import json
from tqdm import tqdm

def generate_filename_from_timestamp(timestamp_ms: int, extension: str, base_dir: str, 
                                   video_name: Optional[str] = None) -> str:
    """
    Generate filename from timestamp matching the frame extractor format
    
    Args:
        timestamp_ms: Timestamp in milliseconds
        extension: File extension (without dot)
        base_dir: Base directory path
        video_name: Optional video name to include in filename
    
    Returns:
        Full file path
    """
    timestamp_str = f"{timestamp_ms:08.0f}ms"
    
    if video_name:
        filename = f"frame_{video_name}_{timestamp_str}.{extension.lower()}"
    else:
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
               video_name: Optional[str] = None,
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
        video_name: Name of the video (for multi-video frame files)
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
    
    # Generate file paths (with or without video name)
    source_path = generate_filename_from_timestamp(timestamp_ms, file_extension, output_frames_dir, video_name)
    destination_path = generate_filename_from_timestamp(timestamp_ms, file_extension, output_dir, video_name)
    
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
            source_filename = os.path.basename(source_path)
            print(f"File '{source_filename}' {operation} to '{category}' directory.")
        
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
                           video_name: Optional[str] = None,
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
        video_name: Name of the video (for multi-video frame files)
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
        error_msg = f"CSV file '{csv_file}' not found"
        if verbose:
            print(f"Error: {error_msg}")
        return {'error': error_msg, 'csv_file': csv_file, 'video_name': video_name}
    except Exception as e:
        error_msg = f"Error reading CSV file '{csv_file}': {e}"
        if verbose:
            print(f"Error: {error_msg}")
        return {'error': error_msg, 'csv_file': csv_file, 'video_name': video_name}
    
    # Validate required columns
    required_columns = [timestamp_column, label_column]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing columns {missing_columns} in {csv_file}"
        if verbose:
            print(f"Error: {error_msg}")
            print(f"Available columns: {list(df.columns)}")
        return {'error': error_msg, 'csv_file': csv_file, 'video_name': video_name}
    
    # Initialize statistics
    stats = {
        'csv_file': csv_file,
        'video_name': video_name,
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
        if video_name:
            print(f"Video: {video_name}")
        print(f"Timestamp column: {timestamp_column}")
        print(f"Label column: {label_column}")
        print(f"Class mapping: {class_names}")
        print(f"Source directory: {output_frames_dir}")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Operation: {'Move' if use_move else 'Copy'}")
        print("-" * 50)
    
    # Process each row with progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {video_name or 'frames'}", disable=not verbose):
        success = move_frame(
            row, timestamp_column, label_column, class_names,
            dataset_dir, output_frames_dir, video_name, file_extension, use_move, False  # Disable verbose for individual operations
        )
        
        if success:
            stats['successful_operations'] += 1
            label_value = row[label_column]
            if label_value in class_names:
                class_name = class_names[label_value]
                stats['class_distribution'][class_name] += 1
        else:
            stats['failed_operations'] += 1
    
    # Print final statistics for this CSV
    if verbose:
        print("-" * 50)
        print(f"Results for {video_name or csv_file}:")
        print(f"  Total rows processed: {stats['total_rows']}")
        print(f"  Successful operations: {stats['successful_operations']}")
        print(f"  Failed operations: {stats['failed_operations']}")
        print(f"  Class distribution:")
        for class_name, count in stats['class_distribution'].items():
            print(f"    {class_name}: {count}")
        if stats['total_rows'] > 0:
            print(f"  Success rate: {(stats['successful_operations']/stats['total_rows']*100):.1f}%")
    
    return stats

class MultiClipFrameOrganizer:
    def __init__(self, dataset_dir: str, output_frames_dir: str, 
                 file_extension: str = "jpg", use_move: bool = True):
        """
        Initialize multi-clip frame organizer
        
        Args:
            dataset_dir: Base dataset directory
            output_frames_dir: Source directory with extracted frames
            file_extension: File extension without dot
            use_move: If True, move files; if False, copy files
        """
        self.dataset_dir = dataset_dir
        self.output_frames_dir = output_frames_dir
        self.file_extension = file_extension
        self.use_move = use_move
        self.results = []
        
        # Create dataset directory
        ensure_directory_exists(dataset_dir)
    
    def organize_from_clip_mapping(self,
                                 clip_csv_mapping: Dict[str, str],
                                 timestamp_column: str,
                                 label_column: str,
                                 class_names: Dict[int, str],
                                 verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Organize frames from multiple clips using a mapping of clip names to CSV files
        
        Args:
            clip_csv_mapping: Dictionary mapping clip names to CSV file paths
            timestamp_column: Name of the timestamp column
            label_column: Name of the binary label column
            class_names: Dictionary mapping label values to directory names
            verbose: If True, print detailed information
        
        Returns:
            List of results for each clip
        """
        if verbose:
            print(f"\U0001f3ac MULTI-CLIP FRAME ORGANIZATION")
            print("="*60)
            print(f"Processing {len(clip_csv_mapping)} clips...")
            print(f"Dataset directory: {self.dataset_dir}")
            print(f"Source frames directory: {self.output_frames_dir}")
            print(f"Class mapping: {class_names}")
            print(f"Operation: {'Move' if self.use_move else 'Copy'}")
            print("="*60)
        
        total_successful = 0
        total_failed = 0
        total_processed = 0
        failed_clips = []
        
        # Process each clip
        for i, (clip_name, csv_file) in enumerate(clip_csv_mapping.items()):
            if verbose:
                print(f"\n\U0001f4f9 [{i+1}/{len(clip_csv_mapping)}] Processing clip: {clip_name}")
                print(f"\U0001f4c4 CSV file: {csv_file}")
            
            # Check if CSV file exists
            if not os.path.exists(csv_file):
                error_msg = f"CSV file not found: {csv_file}"
                if verbose:
                    print(f"\u274c Error: {error_msg}")
                failed_clips.append(clip_name)
                self.results.append({
                    'clip_name': clip_name,
                    'csv_file': csv_file,
                    'error': error_msg
                })
                continue
            
            try:
                # Organize frames for this clip
                result = organize_frames_from_csv(
                    csv_file=csv_file,
                    timestamp_column=timestamp_column,
                    label_column=label_column,
                    class_names=class_names,
                    dataset_dir=self.dataset_dir,
                    output_frames_dir=self.output_frames_dir,
                    video_name=clip_name,
                    file_extension=self.file_extension,
                    use_move=self.use_move,
                    verbose=verbose
                )
                
                if 'error' in result:
                    failed_clips.append(clip_name)
                    if verbose:
                        print(f"\u274c Failed: {result['error']}")
                else:
                    total_successful += result['successful_operations']
                    total_failed += result['failed_operations']
                    total_processed += result['total_rows']
                    
                    if verbose:
                        print(f"\u2705 Success: {result['successful_operations']}/{result['total_rows']} frames")
                
                self.results.append(result)
                
            except Exception as e:
                error_msg = f"Unexpected error processing {clip_name}: {str(e)}"
                if verbose:
                    print(f"\u274c Error: {error_msg}")
                failed_clips.append(clip_name)
                self.results.append({
                    'clip_name': clip_name,
                    'csv_file': csv_file,
                    'error': error_msg
                })
        
        # Print overall summary
        if verbose:
            self._print_summary(
                clip_csv_mapping, failed_clips, total_processed, 
                total_successful, total_failed, class_names
            )
        
        return self.results
    
    def organize_from_csv_list(self,
                             csv_files: List[str],
                             timestamp_column: str,
                             label_column: str,
                             class_names: Dict[int, str],
                             extract_clip_names: bool = True,
                             verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Organize frames from a list of CSV files
        
        Args:
            csv_files: List of CSV file paths
            timestamp_column: Name of the timestamp column
            label_column: Name of the binary label column
            class_names: Dictionary mapping label values to directory names
            extract_clip_names: If True, extract clip names from CSV filenames
            verbose: If True, print detailed information
        
        Returns:
            List of results for each CSV file
        """
        # Create clip mapping from CSV files
        clip_csv_mapping = {}
        for csv_file in csv_files:
            if extract_clip_names:
                # Extract clip name from filename (remove extension and path)
                clip_name = Path(csv_file).stem
                # Remove common suffixes
                for suffix in ['_codes_MLP', '_codes', '_annotations']:
                    clip_name = clip_name.replace(suffix, '')
            else:
                clip_name = None
            
            clip_csv_mapping[clip_name or csv_file] = csv_file
        
        return self.organize_from_clip_mapping(
            clip_csv_mapping, timestamp_column, label_column, class_names, verbose
        )
    
    def _print_summary(self, clip_csv_mapping: Dict[str, str], failed_clips: List[str],
                      total_processed: int, total_successful: int, total_failed: int,
                      class_names: Dict[int, str]):
        """Print comprehensive summary"""
        successful_clips = len(clip_csv_mapping) - len(failed_clips)
        
        print("\n" + "="*60)
        print("\U0001f4ca MULTI-CLIP ORGANIZATION SUMMARY")
        print("="*60)
        print(f"\u2705 Successful clips: {successful_clips}")
        print(f"\u274c Failed clips: {len(failed_clips)}")
        print(f"\U0001f4f8 Total frames processed: {total_processed:,}")
        print(f"\u2705 Successfully organized: {total_successful:,}")
        print(f"\u274c Failed operations: {total_failed:,}")
        
        if total_processed > 0:
            print(f"\U0001f4c8 Overall success rate: {(total_successful/total_processed*100):.1f}%")
        
        # Class distribution summary
        print(f"\n\U0001f4ca COMBINED CLASS DISTRIBUTION:")
        combined_distribution = {class_name: 0 for class_name in class_names.values()}
        for result in self.results:
            if 'class_distribution' in result:
                for class_name, count in result['class_distribution'].items():
                    combined_distribution[class_name] += count
        
        for class_name, count in combined_distribution.items():
            print(f"   {class_name}: {count:,}")
        
        print(f"\n\U0001f4c1 Dataset location: {self.dataset_dir}")
        
        if failed_clips:
            print(f"\n\u274c Failed clips:")
            for clip in failed_clips:
                print(f"   \u2022 {clip}")
        
        print("="*60)
        print("\U0001f3af Multi-clip frame organization complete!")
        print("="*60)
    
    def save_results(self, filepath: str):
        """Save organization results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def get_organized_files_count(self) -> Dict[str, int]:
        """Get count of organized files by class"""
        counts = {}
        for class_dir in os.listdir(self.dataset_dir):
            class_path = os.path.join(self.dataset_dir, class_dir)
            if os.path.isdir(class_path):
                files = [f for f in os.listdir(class_path) if f.endswith(f'.{self.file_extension}')]
                counts[class_dir] = len(files)
        return counts

def organize_frames_from_multiple_clips(clip_csv_mapping: Dict[str, str],
                                      timestamp_column: str,
                                      label_column: str,
                                      class_names: Dict[int, str],
                                      dataset_dir: str,
                                      output_frames_dir: str,
                                      file_extension: str = "jpg",
                                      use_move: bool = True,
                                      verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Simple function to organize frames from multiple clips
    
    Args:
        clip_csv_mapping: Dictionary mapping clip names to CSV file paths
        timestamp_column: Name of the timestamp column
        label_column: Name of the binary label column
        class_names: Dictionary mapping label values to directory names
        dataset_dir: Base dataset directory
        output_frames_dir: Source directory with extracted frames
        file_extension: File extension without dot
        use_move: If True, move files; if False, copy files
        verbose: If True, print detailed information
    
    Returns:
        List of results for each clip
    """
    organizer = MultiClipFrameOrganizer(dataset_dir, output_frames_dir, file_extension, use_move)
    return organizer.organize_from_clip_mapping(
        clip_csv_mapping, timestamp_column, label_column, class_names, verbose
    )

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

# Convenience functions
def organize_face_detection_from_multiple_clips(clip_csv_mapping: Dict[str, str],
                                               dataset_dir: str,
                                               output_frames_dir: str,
                                               timestamp_column: str = 'onset_milliseconds',
                                               label_column: str = 'has_faces',
                                               use_move: bool = True,
                                               verbose: bool = True) -> List[Dict[str, Any]]:
    """Convenience function for face detection from multiple clips"""
    class_names = {0: 'no_face', 1: 'face'}
    return organize_frames_from_multiple_clips(
        clip_csv_mapping, timestamp_column, label_column, class_names,
        dataset_dir, output_frames_dir, "jpg", use_move, verbose
    )

def organize_emotion_from_multiple_clips(clip_csv_mapping: Dict[str, str],
                                       dataset_dir: str,
                                       output_frames_dir: str,
                                       timestamp_column: str = 'onset_milliseconds',
                                       label_column: str = 'is_happy',
                                       positive_emotion: str = 'happy',
                                       negative_emotion: str = 'not_happy',
                                       use_move: bool = True,
                                       verbose: bool = True) -> List[Dict[str, Any]]:
    """Convenience function for emotion detection from multiple clips"""
    class_names = {0: negative_emotion, 1: positive_emotion}
    return organize_frames_from_multiple_clips(
        clip_csv_mapping, timestamp_column, label_column, class_names,
        dataset_dir, output_frames_dir, "jpg", use_move, verbose
    )

if __name__ == '__main__':
    clip_mapping = {
        'clip1_MLP': 'data/clip01/in/clip1_codes_MLP.csv',
        'clip2_AHKJ': 'data/clip02/in/clip2_codes_AHKJ.csv', 
        'clip3_MLP': 'data/clip03/in/clip3_codes_MLP.csv'
    }

    results = organize_emotion_from_multiple_clips(
        clip_csv_mapping=clip_mapping,
        dataset_dir='data/combined/emotion_dataset',
        output_frames_dir='data/combined/frames/',  # Contains frames from all clips
        label_column='c_excite_face',
        positive_emotion='excited',
        negative_emotion='not_excited'
    )
