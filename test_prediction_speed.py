#!/usr/bin/env python
"""
Test script to measure prediction speed for the emotion detection pipeline.
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

def test_prediction_speed():
    """Run prediction and measure timing metrics."""

    # Setup paths
    predict_script = "src/enhanced_pipeline/predict.py"
    models_file = "data/my_results/comprehensive_pipeline_results.json"
    test_video = "data/clip01/in/clip1_MLP.mp4"
    output_file = "test_speed_output.csv"

    # Check if files exist
    if not os.path.exists(predict_script):
        print(f"Error: {predict_script} not found")
        return

    if not os.path.exists(models_file):
        print(f"Error: {models_file} not found")
        return

    if not os.path.exists(test_video):
        print(f"Error: {test_video} not found")
        return

    # Get video info
    print(f"Testing prediction speed on: {test_video}")

    # Get video duration and frame count using ffprobe
    try:
        duration_cmd = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {test_video}"
        duration = float(subprocess.check_output(duration_cmd, shell=True).decode().strip())

        frame_cmd = f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 {test_video}"
        frame_count = int(subprocess.check_output(frame_cmd, shell=True).decode().strip())

        print(f"Video duration: {duration:.2f} seconds")
        print(f"Total frames: {frame_count}")
    except:
        print("Could not get video metadata, continuing anyway...")
        duration = None
        frame_count = None

    # Run prediction with timing
    print("\nRunning prediction...")
    start_time = time.time()

    cmd = [
        "python", predict_script,
        "--models", models_file,
        "--video", test_video,
        "--output", output_file
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        end_time = time.time()

        if result.returncode != 0:
            print(f"Error running prediction: {result.stderr}")
            return

        elapsed_time = end_time - start_time

        # Calculate metrics
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Total prediction time: {elapsed_time:.2f} seconds")

        if duration:
            speed_ratio = duration / elapsed_time
            print(f"Speed ratio: {speed_ratio:.2f}x real-time")

            if speed_ratio > 1:
                print(f"  → Faster than real-time by {speed_ratio:.2f}x")
            else:
                print(f"  → Slower than real-time by {1/speed_ratio:.2f}x")

        if frame_count:
            fps = frame_count / elapsed_time
            print(f"Processing speed: {fps:.2f} frames/second")
            ms_per_frame = (elapsed_time * 1000) / frame_count
            print(f"Time per frame: {ms_per_frame:.2f} ms")

        # Check output file
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                lines = f.readlines()
                print(f"\nOutput file created with {len(lines)} lines")

            # Clean up test output
            os.remove(output_file)
            print(f"Cleaned up test output file: {output_file}")

        # Return metrics for documentation
        metrics = {
            "total_time_seconds": round(elapsed_time, 2),
            "fps": round(fps, 2) if frame_count else None,
            "ms_per_frame": round(ms_per_frame, 2) if frame_count else None,
            "speed_ratio": round(speed_ratio, 2) if duration else None
        }

        return metrics

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Emotion Detection Pipeline - Prediction Speed Test")
    print("-" * 50)

    # Ensure PYTHONPATH is set
    current_dir = os.getcwd()
    src_path = os.path.join(current_dir, "src")
    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = f"{src_path}:{os.environ['PYTHONPATH']}"
    else:
        os.environ["PYTHONPATH"] = src_path

    metrics = test_prediction_speed()

    if metrics:
        print("\n" + "="*50)
        print("SUMMARY FOR DOCUMENTATION:")
        print("="*50)
        print(f"- Total prediction time: {metrics['total_time_seconds']} seconds")
        if metrics['fps']:
            print(f"- Processing speed: {metrics['fps']} frames/second")
        if metrics['ms_per_frame']:
            print(f"- Time per frame: {metrics['ms_per_frame']} ms")
        if metrics['speed_ratio']:
            if metrics['speed_ratio'] > 1:
                print(f"- Performance: {metrics['speed_ratio']}x faster than real-time")
            else:
                print(f"- Performance: {round(1/metrics['speed_ratio'], 2)}x slower than real-time")