import os
import sys
import time
from icrawler.builtin import BingImageCrawler
from PIL import Image
import pandas as pd

def download_emotions(characters_movies, emotions, film_company, download_folder):
    # Create folder for images
    root_dir = download_folder
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    # Use Bing crawler (more reliable than Google for this)
    for character_movie in characters_movies:
        character = character_movie[0]
        movie = character_movie[1]
        for emotion in emotions:
            query = f'"{character}" "{movie}" {emotion} {film_company}'
            print(f"Searching for: {query}")
            
            # Create subfolder for this character/emotion
            folder_name = f"{character.replace(' ', '_')}_{emotion}"
            save_dir = os.path.join(root_dir, folder_name)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Configure and run the crawler
            bing_crawler = BingImageCrawler(
                storage={'root_dir': save_dir},
                downloader_threads=1,
                parser_threads=1
            )
            
            # Download 3 images
            bing_crawler.crawl(keyword=query, max_num=3, file_idx_offset=0)
            
            # Convert images to PNG
            for i, filename in enumerate(os.listdir(save_dir)):
                old_path = os.path.join(save_dir, filename)
                new_path = os.path.join(save_dir, f"image_{i + 1}.png")
                try:
                    with Image.open(old_path) as img:
                        img.convert('RGB').save(new_path, 'PNG')
                    os.remove(old_path)
                    print(f"Converted and saved {filename} as image_{i + 1}.png")
                except Exception as e:
                    print(f"Error converting {filename} to PNG: {e}")
            
            print(f"Downloaded images to {save_dir}")
            
            # Add delay between searches
            time.sleep(2)
            print("-" * 50)

if __name__ == "__main__":
    characters_file = sys.argv[1]
    characters_df = pd.read_csv(characters_file, skiprows=2, quotechar='"')
    characters = list(characters_df['character'])
    movies = list(characters_df['movie'])
    characters_movies = zip(characters, movies)
    emotions_file = sys.argv[2]
    emotions_df = pd.read_csv(emotions_file)
    emotions = list(emotions_df['emotion'])
    film_company = sys.argv[3]
    download_folder = sys.argv[4]
    download_emotions(characters_movies, emotions, film_company, download_folder)
