import cv2
import os
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

def generate_filename_from_timestamp(current_time_ms, image_format, output_dir):
    # Generate filename with timestamp
    timestamp_str = f"{current_time_ms:08.0f}ms"
    filename = f"frame_{timestamp_str}.{image_format.lower()}"
    filepath = os.path.join(output_dir, filename)

    return filepath
    
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

    def extract_frames(self, image_format="jpg", image_quality=95):
        """
        Extract frames from the video at specified intervals
        
        Args:
            image_format (str): Output image format ('jpg', 'png', 'bmp')
            image_quality (int): Image quality for JPEG (1-100)
        """
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Calculate frame interval (how many frames to skip)
        frame_interval = int((self.interval_ms / 1000) * self.fps)
        
        # Calculate total number of frames to extract
        total_extractions = int(self.duration_ms / self.interval_ms)
        
        print(f"Extracting approximately {total_extractions} frames...")
        print(f"Frame interval: every {frame_interval} frames")
        
        extracted_count = 0
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
            # Set video position to current time
            self.cap.set(cv2.CAP_PROP_POS_MSEC, current_time_ms)
            
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Generate filename with timestamp
            filepath = generate_filename_from_timestamp(current_time_ms, image_format, self.output_dir)
            
            # Save frame
            success = cv2.imwrite(filepath, frame, encode_params)
            
            if success:
                extracted_count += 1
                pbar.update(1)
            else:
                print(f"Failed to save frame at {current_time_ms}ms")
            
            # Move to next interval
            current_time_ms += self.interval_ms
        
        pbar.close()
        self.cap.release()
        
        print(f"\nExtraction complete!")
        print(f"Extracted {extracted_count} frames to '{self.output_dir}'")
        
        return extracted_count
    
    def extract_frames_by_frame_number(self, image_format="jpg", image_quality=95):
        """
        Alternative method: Extract frames by iterating through frame numbers
        This method can be more precise for some videos
        """
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        
        # Calculate frame interval
        frame_interval = int((self.interval_ms / 1000) * self.fps)
        
        print(f"Extracting frames every {frame_interval} frames...")
        
        extracted_count = 0
        frame_number = 0
        
        # Set up image encoding parameters
        encode_params = []
        if image_format.lower() == 'jpg' or image_format.lower() == 'jpeg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, image_quality]
        elif image_format.lower() == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        
        # Progress bar
        pbar = tqdm(total=self.total_frames//frame_interval, desc="Extracting frames")
        
        while True:
            # Set frame position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Calculate timestamp
            timestamp_ms = (frame_number / self.fps) * 1000
            
            # Generate filename
            timestamp_str = f"{timestamp_ms:08.0f}ms"
            filename = f"frame_{frame_number:06d}_{timestamp_str}.{image_format.lower()}"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save frame
            success = cv2.imwrite(filepath, frame, encode_params)
            
            if success:
                extracted_count += 1
                pbar.update(1)
            else:
                print(f"Failed to save frame {frame_number}")
            
            # Move to next frame
            frame_number += frame_interval
            
            if frame_number >= self.total_frames:
                break
        
        pbar.close()
        self.cap.release()
        
        print(f"\nExtraction complete!")
        print(f"Extracted {extracted_count} frames to '{self.output_dir}'")
        
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

def extract_frames_simple(video_path, output_dir="frames", interval_ms=100, image_format="jpg"):
    """
    Simple function to extract frames from video
    
    Args:
        video_path (str): Path to the MP4 file
        output_dir (str): Output directory for frames
        interval_ms (int): Interval in milliseconds
        image_format (str): Image format (jpg, png, bmp)
    """
    extractor = FrameExtractor(video_path, output_dir, interval_ms)
    return extractor.extract_frames(image_format=image_format)

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
    
    args = parser.parse_args()
    
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
            extracted_count = extractor.extract_frames(args.format, args.quality)
        else:
            extracted_count = extractor.extract_frames_by_frame_number(args.format, args.quality)
        
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
