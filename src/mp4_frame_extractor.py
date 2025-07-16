import cv2
import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Dict, Any, Optional
import json
import time

def generate_filename_from_timestamp(current_time_ms: int, image_format: str, output_dir: str, 
                                   video_name: Optional[str] = None) -> str:
    """
    Generate filename with timestamp and optional video name
    
    Args:
        current_time_ms: Timestamp in milliseconds
        image_format: File extension (without dot)
        output_dir: Output directory path
        video_name: Optional video name to include in filename
    
    Returns:
        Full file path
    """
    timestamp_str = f"{current_time_ms:08.0f}ms"
    
    if video_name:
        filename = f"frame_{video_name}_{timestamp_str}.{image_format.lower()}"
    else:
        filename = f"frame_{timestamp_str}.{image_format.lower()}"
    
    filepath = os.path.join(output_dir, filename)
    return filepath

def is_frame_black(frame, threshold=10):
    """
    Check if a frame is mostly black
    
    Args:
        frame: OpenCV frame (BGR format)
        threshold: Average pixel value threshold (0-255)
    
    Returns:
        bool: True if frame is mostly black
    """
    if frame is None:
        return True
    
    # Calculate average pixel value
    avg_pixel_value = np.mean(frame)
    return avg_pixel_value < threshold

def is_frame_valid(frame, min_brightness=5, min_variance=10):
    """
    Check if a frame is valid (not black, corrupted, or too dark)
    
    Args:
        frame: OpenCV frame (BGR format)
        min_brightness: Minimum average brightness (0-255)
        min_variance: Minimum variance in pixel values
    
    Returns:
        bool: True if frame is valid
    """
    if frame is None or frame.size == 0:
        return False
    
    # Check brightness
    avg_brightness = np.mean(frame)
    if avg_brightness < min_brightness:
        return False
    
    # Check variance (to detect uniform/blank frames)
    variance = np.var(frame)
    if variance < min_variance:
        return False
    
    return True

class FrameExtractor:
    def __init__(self, video_path: str, output_dir: str = "extracted_frames", 
                 interval_ms: int = 100, include_video_name: bool = True):
        """
        Initialize the frame extractor
        
        Args:
            video_path: Path to the input MP4 file
            output_dir: Directory to save extracted frames
            interval_ms: Interval in milliseconds between extracted frames
            include_video_name: Whether to include video name in frame filenames
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.interval_ms = interval_ms
        self.include_video_name = include_video_name
        
        # Extract video name (without extension)
        self.video_name = Path(video_path).stem if include_video_name else None
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Video capture object
        self.cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_ms = (self.total_frames / self.fps) * 1000
        
    def get_video_info(self) -> Dict[str, Any]:
        """Get video information"""
        if not self.cap.isOpened():
            return {'error': f"Cannot open video file: {self.video_path}"}
        
        # Read first frame to get dimensions
        ret, frame = self.cap.read()
        if ret:
            height, width, channels = frame.shape
        else:
            height = width = channels = None
        
        return {
            'video_path': self.video_path,
            'video_name': self.video_name,
            'width': width,
            'height': height,
            'channels': channels,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration_ms': self.duration_ms,
            'duration_seconds': self.duration_ms / 1000 if self.duration_ms else 0
        }
    
    def print_video_info(self):
        """Print video information"""
        info = self.get_video_info()
        if 'error' in info:
            print(f"Error: {info['error']}")
            return
        
        print(f"Video: {info['video_name'] or 'Unknown'}")
        print(f"  Path: {info['video_path']}")
        print(f"  FPS: {info['fps']}")
        print(f"  Total frames: {info['total_frames']}")
        print(f"  Duration: {info['duration_ms']:.2f} ms ({info['duration_seconds']:.2f} seconds)")
        print(f"  Resolution: {info['width']}x{info['height']} pixels, {info['channels']} channels")
        print(f"  Extraction interval: {self.interval_ms} ms")

    def extract_frames(self, image_format: str = "jpg", image_quality: int = 95, 
                      skip_black_frames: bool = True, retry_on_black: bool = True, 
                      max_retries: int = 5) -> Dict[str, Any]:
        """
        Extract frames from the video at specified intervals
        
        Args:
            image_format: Output image format ('jpg', 'png', 'bmp')
            image_quality: Image quality for JPEG (1-100)
            skip_black_frames: Skip frames that are mostly black
            retry_on_black: Retry with slight offset if black frame detected
            max_retries: Maximum retry attempts for each timestamp
        
        Returns:
            Dictionary with extraction results
        """
        if not self.cap.isOpened():
            return {'error': f"Cannot open video file: {self.video_path}"}
        
        # Calculate frame interval and total extractions
        frame_interval = int((self.interval_ms / 1000) * self.fps)
        total_extractions = int(self.duration_ms / self.interval_ms)
        
        extracted_count = 0
        skipped_count = 0
        current_time_ms = 0
        
        # Set up image encoding parameters
        encode_params = []
        if image_format.lower() in ['jpg', 'jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
        elif image_format.lower() == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        
        # Progress bar
        desc = f"Extracting frames from {self.video_name or 'video'}"
        pbar = tqdm(total=total_extractions, desc=desc)
        
        while current_time_ms < self.duration_ms:
            frame_saved = False
            
            for retry in range(max_retries + 1):
                # Calculate time offset for retry
                time_offset = retry * (self.interval_ms / (max_retries + 2))
                seek_time = current_time_ms + time_offset
                
                # Set video position to current time
                self.cap.set(cv2.CAP_PROP_POS_MSEC, seek_time)
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Check if frame is valid
                if skip_black_frames and not is_frame_valid(frame):
                    if retry < max_retries and retry_on_black:
                        continue  # Try again with offset
                    else:
                        skipped_count += 1
                        break  # Skip this frame
                
                # Frame is valid, save it
                filepath = generate_filename_from_timestamp(
                    current_time_ms, image_format, self.output_dir, self.video_name
                )
                success = cv2.imwrite(filepath, frame, encode_params)
                
                if success:
                    extracted_count += 1
                    frame_saved = True
                    break
                else:
                    print(f"Failed to save frame at {current_time_ms}ms")
                    break
            
            if frame_saved or not retry_on_black:
                pbar.update(1)
            
            # Move to next interval
            current_time_ms += self.interval_ms
        
        pbar.close()
        self.cap.release()
        
        return {
            'video_path': self.video_path,
            'video_name': self.video_name,
            'extracted_count': extracted_count,
            'skipped_count': skipped_count,
            'total_expected': total_extractions,
            'success_rate': (extracted_count / total_extractions * 100) if total_extractions > 0 else 0
        }

class MultiVideoFrameExtractor:
    def __init__(self, output_dir: str = "extracted_frames", interval_ms: int = 100,
                 include_video_name: bool = True):
        """
        Initialize the multi-video frame extractor
        
        Args:
            output_dir: Directory to save all extracted frames
            interval_ms: Interval in milliseconds between extracted frames
            include_video_name: Whether to include video name in frame filenames
        """
        self.output_dir = output_dir
        self.interval_ms = interval_ms
        self.include_video_name = include_video_name
        self.results = []
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def extract_from_videos(self, video_paths: List[str], image_format: str = "jpg",
                           image_quality: int = 95, skip_black_frames: bool = True,
                           verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Extract frames from multiple videos
        
        Args:
            video_paths: List of paths to video files
            image_format: Output image format
            image_quality: Image quality for JPEG
            skip_black_frames: Skip black/invalid frames
            verbose: Print detailed information
        
        Returns:
            List of extraction results for each video
        """
        if verbose:
            print(f"Processing {len(video_paths)} videos...")
            print(f"Output directory: {self.output_dir}")
            print(f"Extraction interval: {self.interval_ms} ms")
            print(f"Include video names: {self.include_video_name}")
            print("-" * 60)
        
        total_extracted = 0
        total_skipped = 0
        successful_videos = 0
        failed_videos = []
        
        for i, video_path in enumerate(video_paths):
            if verbose:
                print(f"\n[{i+1}/{len(video_paths)}] Processing: {Path(video_path).name}")
            
            # Check if video file exists
            if not os.path.exists(video_path):
                error_msg = f"Video file not found: {video_path}"
                if verbose:
                    print(f"Error: {error_msg}")
                failed_videos.append(video_path)
                self.results.append({
                    'video_path': video_path,
                    'error': error_msg
                })
                continue
            
            try:
                # Create extractor for this video
                extractor = FrameExtractor(
                    video_path, self.output_dir, self.interval_ms, self.include_video_name
                )
                
                # Print video info if verbose
                if verbose:
                    extractor.print_video_info()
                
                # Extract frames
                result = extractor.extract_frames(
                    image_format, image_quality, skip_black_frames
                )
                
                if 'error' in result:
                    if verbose:
                        print(f"Error: {result['error']}")
                    failed_videos.append(video_path)
                else:
                    total_extracted += result['extracted_count']
                    total_skipped += result['skipped_count']
                    successful_videos += 1
                    
                    if verbose:
                        print(f"Extracted {result['extracted_count']} frames")
                        if skip_black_frames and result['skipped_count'] > 0:
                            print(f"Skipped {result['skipped_count']} black/invalid frames")
                        print(f"Success rate: {result['success_rate']:.1f}%")
                
                self.results.append(result)
                
            except Exception as e:
                error_msg = f"Unexpected error processing {video_path}: {str(e)}"
                if verbose:
                    print(f"Error: {error_msg}")
                failed_videos.append(video_path)
                self.results.append({
                    'video_path': video_path,
                    'error': error_msg
                })
        
        # Print summary
        if verbose:
            self._print_summary(successful_videos, failed_videos, total_extracted, total_skipped)
        
        return self.results
    
    def _print_summary(self, successful_videos: int, failed_videos: List[str], 
                      total_extracted: int, total_skipped: int):
        """Print extraction summary"""
        print("\n" + "="*60)
        print("\U0001f4ca EXTRACTION SUMMARY")
        print("="*60)
        print(f"\u2705 Successful videos: {successful_videos}")
        print(f"\u274c Failed videos: {len(failed_videos)}")
        print(f"\U0001f4f8 Total frames extracted: {total_extracted:,}")
        if total_skipped > 0:
            print(f"\u26a0\ufe0f  Total frames skipped: {total_skipped:,}")
        print(f"\U0001f4c1 Output directory: {self.output_dir}")
        
        if failed_videos:
            print(f"\n\u274c Failed videos:")
            for video in failed_videos:
                print(f"   \u2022 {video}")
        
        print("="*60)
        print("\U0001f3ac Frame extraction complete!")
        print("="*60)
    
    def save_results(self, filepath: str):
        """Save extraction results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def get_extracted_files(self) -> List[str]:
        """Get list of all extracted frame files"""
        files = []
        for root, dirs, filenames in os.walk(self.output_dir):
            for filename in filenames:
                if filename.startswith('frame_') and any(filename.endswith(ext) for ext in ['.jpg', '.png', '.bmp']):
                    files.append(os.path.join(root, filename))
        return sorted(files)

def extract_frames_from_videos(video_paths: List[str], output_dir: str = "extracted_frames",
                             interval_ms: int = 100, image_format: str = "jpg",
                             include_video_name: bool = True, skip_black_frames: bool = True,
                             verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Simple function to extract frames from multiple videos
    
    Args:
        video_paths: List of paths to video files
        output_dir: Output directory for frames
        interval_ms: Interval in milliseconds
        image_format: Image format (jpg, png, bmp)
        include_video_name: Include video name in filenames
        skip_black_frames: Skip black/invalid frames
        verbose: Print detailed information
    
    Returns:
        List of extraction results
    """
    extractor = MultiVideoFrameExtractor(output_dir, interval_ms, include_video_name)
    return extractor.extract_from_videos(
        video_paths, image_format, skip_black_frames=skip_black_frames, verbose=verbose
    )

def find_video_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Find all video files in a directory
    
    Args:
        directory: Directory to search
        extensions: List of video extensions to look for
    
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                video_files.append(os.path.join(root, file))
    
    return sorted(video_files)

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Extract frames from MP4 videos at regular intervals")
    
    # Input options
    parser.add_argument("videos", nargs="+", help="Paths to video files or directories containing videos")
    parser.add_argument("-o", "--output", default="extracted_frames", 
                       help="Output directory (default: extracted_frames)")
    parser.add_argument("-i", "--interval", type=int, default=100,
                       help="Interval in milliseconds (default: 100)")
    
    # Output options
    parser.add_argument("-f", "--format", default="jpg", choices=["jpg", "png", "bmp"],
                       help="Image format (default: jpg)")
    parser.add_argument("-q", "--quality", type=int, default=95,
                       help="JPEG quality 1-100 (default: 95)")
    
    # Processing options
    parser.add_argument("--no-video-name", action="store_true",
                       help="Don't include video name in frame filenames")
    parser.add_argument("--allow-black-frames", action="store_true",
                       help="Don't skip black/invalid frames")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress detailed output")
    
    # File discovery
    parser.add_argument("--recursive", action="store_true",
                       help="Search directories recursively for video files")
    parser.add_argument("--extensions", nargs="+", default=[".mp4", ".avi", ".mov"],
                       help="Video file extensions to look for (default: .mp4 .avi .mov)")
    
    # Output options
    parser.add_argument("--save-results", 
                       help="Save extraction results to JSON file")
    
    args = parser.parse_args()
    
    # Collect video files
    video_files = []
    for path in args.videos:
        if os.path.isfile(path):
            video_files.append(path)
        elif os.path.isdir(path):
            if args.recursive:
                found_videos = find_video_files(path, args.extensions)
                video_files.extend(found_videos)
                if not args.quiet:
                    print(f"Found {len(found_videos)} video files in {path}")
            else:
                # Only look in immediate directory
                for file in os.listdir(path):
                    filepath = os.path.join(path, file)
                    if os.path.isfile(filepath) and any(file.lower().endswith(ext) for ext in args.extensions):
                        video_files.append(filepath)
        else:
            print(f"Warning: Path not found: {path}")
    
    if not video_files:
        print("Error: No video files found!")
        return
    
    if not args.quiet:
        print(f"Processing {len(video_files)} video files...")
    
    # Extract frames
    try:
        results = extract_frames_from_videos(
            video_files,
            output_dir=args.output,
            interval_ms=args.interval,
            image_format=args.format,
            include_video_name=not args.no_video_name,
            skip_black_frames=not args.allow_black_frames,
            verbose=not args.quiet
        )
        
        # Save results if requested
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.save_results}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")

# Example usage
if __name__ == "__main__":
    # Example 1: Process multiple specific videos
    video_list = [
        "data/clip01/in/clip1_MLP.mp4",
        "data/clip02/in/clip2_AHKJ.mp4",
        "data/clip03/in/clip3_MLP.mp4"
    ]
    
    # This will create files like:
    # frame_clip1_MLP_00000000ms.jpg
    # frame_clip1_MLP_00000100ms.jpg
    # frame_clip2_MLP_00000000ms.jpg
    # etc.
    
    results = extract_frames_from_videos(
        video_list,
        output_dir="data/all_clips/frames",
        interval_ms=100,
        image_format="jpg",
        include_video_name=True,
        skip_black_frames=True
    )
    
    # Example 2: Process all videos in a directory
    # video_dir_files = find_video_files("data/videos/")
    # results = extract_frames_from_videos(video_dir_files, "extracted_frames")
    
    # Example 3: Command line interface
    # main()
