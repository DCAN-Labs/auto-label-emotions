import os
import time
from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

def download_pixar_emotions():
    # Install icrawler if not installed
    try:
        import icrawler
    except ImportError:
        import subprocess
        print("Installing icrawler...")
        subprocess.check_call(["pip", "install", "icrawler"])
    
    # Define Pixar characters and emotions
    characters = ["Joy Inside Out", "Sadness Inside Out", "Anger Inside Out", 
                 "Fear Inside Out", "Disgust Inside Out", "Woody Toy Story"]
    
    emotions = ["happy", "sad", "angry", "surprised"]
    
    # Create folder for images
    root_dir = "pixar_emotions"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # Use Bing crawler (more reliable than Google for this)
    for character in characters:
        for emotion in emotions:
            query = f"{character} {emotion} pixar"
            print(f"Searching for: {query}")
            
            # Create subfolder for this character/emotion
            folder_name = f"{character.replace(' ', '_')}_{emotion}"
            save_dir = os.path.join(root_dir, folder_name)
            
            # Configure and run the crawler
            bing_crawler = BingImageCrawler(
                storage={'root_dir': save_dir},
                downloader_threads=1,
                parser_threads=1
            )
            
            # Download 3 images
            bing_crawler.crawl(keyword=query, max_num=3)
            print(f"Downloaded images to {save_dir}")
            
            # Add delay between searches
            time.sleep(2)
            print("-" * 50)

if __name__ == "__main__":
    download_pixar_emotions()
    