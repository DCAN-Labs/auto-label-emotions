import os
import sys
import time
from icrawler.builtin import BingImageCrawler
from PIL import Image
import pandas as pd

def download_emotions(characters_movies, emotions, film_company, download_folder):
    base_delay = 5  # Initial delay in seconds
    max_retries = 3  # Maximum number of retry attempts

    # Create folder for images
    root_dir = download_folder
    try:
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
    except OSError as e:
        print(f"Error creating root directory {root_dir}: {e}")
        return
    
    # Create a single crawler instance to reuse
    bing_crawler = BingImageCrawler(
        downloader_threads=1,
        parser_threads=1
    )
    
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
            
            try:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
            except OSError as e:
                print(f"Error creating directory {save_dir}: {e}")
                continue
            
            # Update the storage location for this search
            bing_crawler.storage = {'root_dir': save_dir}
            
            # Use the crawler
            try:
                # Inside the loop, replace the crawler use with:
                retry_count = 0
                success = False

                while not success and retry_count < max_retries:
                    try:
                        bing_crawler.crawl(keyword=query, max_num=3, file_idx_offset=0)
                        success = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            print(f"Failed after {max_retries} attempts: {e}")
                            break
                        
                        # Exponential backoff
                        retry_delay = base_delay * (2 ** retry_count) + random.uniform(0, 1)
                        print(f"Search failed, retrying in {retry_delay:.1f} seconds... (Attempt {retry_count+1}/{max_retries})")
                        time.sleep(retry_delay)
            except Exception as e:
                print(f"Error during image crawling for {query}: {e}")
                continue
            
            # Convert images to PNG
            try:
                files = os.listdir(save_dir)
            except OSError as e:
                print(f"Error listing directory {save_dir}: {e}")
                continue
                
            for i, filename in enumerate(files):
                old_path = os.path.join(save_dir, filename)
                new_path = os.path.join(save_dir, f"image_{i + 1}.png")
                try:
                    with Image.open(old_path) as img:
                        img.convert('RGB').save(new_path, 'PNG')
                    try:
                        os.remove(old_path)
                        print(f"Converted and saved {filename} as image_{i + 1}.png")
                    except OSError as e:
                        print(f"Error removing original file {old_path}: {e}")
                except (IOError, OSError) as e:
                    print(f"Error converting {filename} to PNG: {e}")
            
            print(f"Downloaded images to {save_dir}")
            

if __name__ == "__main__":
    # Check for correct number of arguments
    if len(sys.argv) != 5:
        print("Usage: python script.py characters_file emotions_file film_company download_folder")
        sys.exit(1)
        
    characters_file = sys.argv[1]
    emotions_file = sys.argv[2]
    film_company = sys.argv[3]
    download_folder = sys.argv[4]
    
    # Check if files exist
    if not os.path.isfile(characters_file):
        print(f"Error: Characters file '{characters_file}' does not exist")
        sys.exit(1)
    
    if not os.path.isfile(emotions_file):
        print(f"Error: Emotions file '{emotions_file}' does not exist")
        sys.exit(1)
    
    try:
        # Read character data
        characters_df = pd.read_csv(characters_file, skiprows=2, quotechar='"')
        if 'character' not in characters_df.columns or 'movie' not in characters_df.columns:
            print(f"Error: Characters file must contain 'character' and 'movie' columns")
            sys.exit(1)
            
        characters = list(characters_df['character'])
        movies = list(characters_df['movie'])
        characters_movies = list(zip(characters, movies))
        
        # Read emotions data
        emotions_df = pd.read_csv(emotions_file)
        if 'emotion' not in emotions_df.columns:
            print(f"Error: Emotions file must contain an 'emotion' column")
            sys.exit(1)
            
        emotions = list(emotions_df['emotion'])
        
        # Execute main function
        download_emotions(characters_movies, emotions, film_company, download_folder)
        
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)