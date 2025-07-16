import cv2
import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def generate_filename_from_timestamp(current_time_ms, image_format, output_dir):
    """Generate filename with timestamp"""
    timestamp_str = f"{current_time_ms:08.0f}ms"
    filename = f"frame_{timestamp_str}.{image_format.lower()}"
    filepath = os.path.join(output_dir, filename)
    return filepath

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
    def __init__(self, video_path, output_dir="extracted_frames", interval_ms=100):
        """
        Initialize the frame extractor
        
        Args:
            video_path (str): Path to the input MP4 file
            output_dir (str): Directory to save extracted frames
            interval_ms (int): Interval in milliseconds between extracted frames
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.interval_ms = interval_ms
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Video capture object
        self.cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_ms = (self.total_frames / self.fps) * 1000
        
        print(f"Video properties:")
        print(f"  FPS: {self.fps}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  Duration: {self.duration_ms:.2f} ms ({self.duration_ms/1000:.2f} seconds)")
        print(f"  Extraction interval: {self.interval_ms} ms")

    def extract_frames(self, image_format="jpg", image_quality=95, skip_black_frames=True, 
                    retry_on_black=True, max_retries=5):
        """
        Extract frames from the video at specified intervals
        
        Args:
            image_format (str): Output image format ('jpg', 'png', 'bmp')
            image_quality (int): Image quality for JPEG (1-100)
            skip_black_frames (bool): Skip frames that are mostly black
            retry_on_black (bool): Retry with slight offset if black frame detected
            max_retries (int): Maximum retry attempts for each timestamp
        """
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Calculate frame interval (how many frames to skip)
        frame_interval = int((self.interval_ms / 1000) * self.fps)
        
        # Calculate total number of frames to extract
        total_extractions = int(self.duration_ms / self.interval_ms)
        
        print(f"Extracting approximately {total_extractions} frames...")
        print(f"Frame interval: every {frame_interval} frames")
        print(f"Skip black frames: {skip_black_frames}")
        print(f"Retry on black frames: {retry_on_black}")
        
        extracted_count = 0
        skipped_count = 0
        current_time_ms = 0
        
        # Set up image encoding parameters
        encode_params = []
        if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
        elif image_format.lower() == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        
        # Progress bar
        pbar = tqdm(total=total_extractions, desc="Extracting frames")
        
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
                filepath = generate_filename_from_timestamp(current_time_ms, image_format, self.output_dir)
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
        
        print(f"\nExtraction complete!")
        print(f"Extracted {extracted_count} frames to '{self.output_dir}'")
        if skip_black_frames:
            print(f"Skipped {skipped_count} black/invalid frames")
        
        return extracted_count
    
    def extract_frames_by_frame_number(self, image_format="jpg", image_quality=95, 
                                      skip_black_frames=True):
        """
        Alternative method: Extract frames by iterating through frame numbers
        This method can be more precise and avoid seeking issues
        """
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Calculate frame interval
        frame_interval = int((self.interval_ms / 1000) * self.fps)
        
        print(f"Extracting frames every {frame_interval} frames...")
        print(f"Skip black frames: {skip_black_frames}")
        
        extracted_count = 0
        skipped_count = 0
        frame_number = 0
        
        # Set up image encoding parameters
        encode_params = []
        if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
        elif image_format.lower() == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        
        # Progress bar
        pbar = tqdm(total=self.total_frames//frame_interval, desc="Extracting frames")
        
        while frame_number < self.total_frames:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Calculate timestamp
            timestamp_ms = (frame_number / self.fps) * 1000
            
            # Check if frame is valid
            if skip_black_frames and not is_frame_valid(frame):
                skipped_count += 1
                pbar.update(1)
                frame_number += frame_interval
                continue
            
            # Generate filename
            filepath = generate_filename_from_timestamp(timestamp_ms, image_format, self.output_dir)
            
            # Save frame
            success = cv2.imwrite(filepath, frame, encode_params)
            
            if success:
                extracted_count += 1
                pbar.update(1)
            else:
                print(f"Failed to save frame {frame_number}")
            
            # Move to next frame
            frame_number += frame_interval
        
        pbar.close()
        self.cap.release()
        
        print(f"\nExtraction complete!")
        print(f"Extracted {extracted_count} frames to '{self.output_dir}'")
        if skip_black_frames:
            print(f"Skipped {skipped_count} black/invalid frames")
        
        return extracted_count
    
    def get_frame_info(self):
        """Get information about the video frames"""
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Read first frame to get dimensions
        ret, frame = self.cap.read()
        if ret:
            height, width, channels = frame.shape
            print(f"Frame dimensions: {width}x{height} pixels, {channels} channels")
        
        self.cap.release()
        
        return {
            'width': width if ret else None,
            'height': height if ret else None,
            'channels': channels if ret else None,
            'fps': self.fps,
            'total_frames': self.total_frames,
            'duration_ms': self.duration_ms
        }

def extract_frames_simple(video_path, output_dir="frames", interval_ms=100, image_format="jpg", 
                         skip_black_frames=True):
    """
    Simple function to extract frames from video
    
    Args:
        video_path (str): Path to the MP4 file
        output_dir (str): Output directory for frames
        interval_ms (int): Interval in milliseconds
        image_format (str): Image format (jpg, png, bmp)
        skip_black_frames (bool): Skip black/invalid frames
    """
    extractor = FrameExtractor(video_path, output_dir, interval_ms)
    return extractor.extract_frames(image_format=image_format, skip_black_frames=skip_black_frames)

def clean_existing_black_frames(frames_dir, delete_black_frames=False):
    """
    Clean existing extracted frames by identifying and optionally removing black frames
    
    Args:
        frames_dir (str): Directory containing extracted frames
        delete_black_frames (bool): If True, delete black frames; if False, just report them
    
    Returns:
        list: List of black frame files found
    """
    black_frames = []
    
    if not os.path.exists(frames_dir):
        print(f"Directory {frames_dir} does not exist")
        return black_frames
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for file in os.listdir(frames_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"Checking {len(image_files)} frames for black/invalid frames...")
    
    for filename in tqdm(image_files, desc="Checking frames"):
        filepath = os.path.join(frames_dir, filename)
        
        # Read frame
        frame = cv2.imread(filepath)
        
        if not is_frame_valid(frame):
            black_frames.append(filename)
            
            if delete_black_frames:
                try:
                    os.remove(filepath)
                    print(f"Deleted black frame: {filename}")
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")
    
    print(f"\nFound {len(black_frames)} black/invalid frames")
    if delete_black_frames:
        print(f"Deleted {len(black_frames)} black frames")
    else:
        print("To delete these frames, run with delete_black_frames=True")
    
    return black_frames

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Extract frames from MP4 video at regular intervals")
    parser.add_argument("video_path", help="Path to the input MP4 file")
    parser.add_argument("-o", "--output", default="extracted_frames", 
                       help="Output directory (default: extracted_frames)")
    parser.add_argument("-i", "--interval", type=int, default=100,
                       help="Interval in milliseconds (default: 100)")
    parser.add_argument("-f", "--format", default="jpg", choices=["jpg", "png", "bmp"],
                       help="Image format (default: jpg)")
    parser.add_argument("-q", "--quality", type=int, default=95,
                       help="JPEG quality 1-100 (default: 95)")
    parser.add_argument("--method", choices=["time", "frame"], default="time",
                       help="Extraction method: by time or frame number (default: time)")
    parser.add_argument("--skip-black", action="store_true", default=True,
                       help="Skip black/invalid frames (default: True)")
    parser.add_argument("--retry-black", action="store_true", default=True,
                       help="Retry with offset if black frame detected (default: True)")
    parser.add_argument("--clean-existing", action="store_true",
                       help="Clean existing frames directory of black frames")
    
    args = parser.parse_args()
    
    # Clean existing frames if requested
    if args.clean_existing:
        if os.path.exists(args.output):
            black_frames = clean_existing_black_frames(args.output, delete_black_frames=True)
            print(f"Cleaned {len(black_frames)} black frames from {args.output}")
        return
    
    # Check if video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found!")
        return
    
    # Create extractor
    extractor = FrameExtractor(args.video_path, args.output, args.interval)
    
    # Show video info
    extractor.get_frame_info()
    
    # Extract frames
    try:
        if args.method == "time":
            extracted_count = extractor.extract_frames(
                args.format, args.quality, 
                skip_black_frames=args.skip_black,
                retry_on_black=args.retry_black
            )
        else:
            extracted_count = extractor.extract_frames_by_frame_number(
                args.format, args.quality,
                skip_black_frames=args.skip_black
            )
        
        print(f"\nSuccess! Extracted {extracted_count} frames.")
        
    except Exception as e:
        print(f"Error during extraction: {e}")

# Example usage
if __name__ == "__main__":
    # Example 1: Simple usage
    extract_frames_simple("data/clip01/in/clip1_MLP.mp4", "data/clip01/out/output_frames", 100, "jpg")
    
    # Example 2: Advanced usage
    # extractor = FrameExtractor("input_video.mp4", "frames_100ms", 100)
    # extractor.get_frame_info()
    # extractor.extract_frames(image_format="png", image_quality=95)
    
    # Example 3: Command line interface
    # main()
