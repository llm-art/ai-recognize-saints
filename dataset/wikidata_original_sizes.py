import os
import pandas as pd
import requests
from tqdm import tqdm
import json
from PIL import Image
import io
import numpy as np
import logging

# Increase PIL threshold to match our max_pixels value
Image.MAX_IMAGE_PIXELS = 178956970

# Set up logging
logging.basicConfig(
    filename='image_download.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create the necessary directories if they don't exist
wikidata_dir = os.path.join(os.getcwd(), 'wikidata')
wikidata_data_dir = os.path.join(os.getcwd(), 'wikidata-data')
# Change the output directory to the requested path
jpeg_images_dir = os.path.join('/home/ubuntu/gspinaci/LLM-test/dataset/wikidata/JPEGImages')
os.makedirs(wikidata_dir, exist_ok=True)
os.makedirs(wikidata_data_dir, exist_ok=True)
os.makedirs(jpeg_images_dir, exist_ok=True)

logging.info(f"Output directory: {jpeg_images_dir}")

# Read the CSV file with the filtered images
images_df = pd.read_csv(os.path.join(wikidata_dir, 'wikidata.csv'))
logging.info(f"Loaded {len(images_df)} images from CSV")

# Read the top classes
top_classes_df = pd.read_csv(os.path.join(wikidata_data_dir, 'pre_classes.csv'))
iconclass_counts = pd.Series(index=top_classes_df['iconclass'].values, data=top_classes_df['count'].values)
logging.info(f"Loaded {len(iconclass_counts)} top classes")

# Initialize lists to store the image data and image sizes
image_data = []
image_sizes = []
failed_downloads = []

# Enhanced function to download an image from a URL while preserving original size and aspect ratio
def download_image(url, save_path, max_pixels=178956970, target_size=None):
    """
    Download an image from a URL, preserving original size and aspect ratio.
    
    Args:
        url: URL of the image to download
        save_path: Path where the image will be saved
        max_pixels: Maximum number of pixels allowed (width × height)
        target_size: Optional (width, height) tuple for resizing (set to None to preserve original size)
    
    Returns:
        tuple: (success, width, height) - success is a boolean, width and height are the dimensions of the saved image
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    try:
        # Download the image
        logging.info(f"Downloading {url}")
        response = requests.get(url, headers=headers, stream=True)
        if response.status_code != 200:
            logging.error(f"Failed to download {url}: HTTP status code {response.status_code}")
            return False, 0, 0
            
        # Read the image data into memory
        image_data = io.BytesIO()
        for chunk in response.iter_content(1024):
            image_data.write(chunk)
        image_data.seek(0)
        
        # Open the image
        with Image.open(image_data) as img:
            # Convert to RGB if needed (handles PNG, RGBA, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            width, height = img.size
            num_pixels = width * height
            original_size = f"{width}x{height}"
            logging.info(f"Original image size: {original_size} ({num_pixels} pixels)")
            
            # Check if resizing is needed to prevent decompression bomb
            if num_pixels > max_pixels:
                # Calculate new dimensions while maintaining aspect ratio
                ratio = width / height
                if ratio > 1:
                    new_width = int(np.sqrt(max_pixels * ratio))
                    new_height = int(new_width / ratio)
                else:
                    new_height = int(np.sqrt(max_pixels / ratio))
                    new_width = int(new_height * ratio)
                
                # Resize the image
                img = img.resize((new_width, new_height), Image.LANCZOS)
                width, height = new_width, new_height
                logging.info(f"Resized image from {original_size} to {width}x{height}")
            
            # Save the image
            img.save(save_path, 'JPEG', quality=95)
            
            # Verify the image was saved successfully
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                if file_size > 0:
                    logging.info(f"Successfully saved {save_path} ({file_size} bytes)")
                    return True, width, height
                else:
                    logging.warning(f"File saved but has zero size: {save_path}")
                    os.remove(save_path)  # Remove empty file
                    return False, 0, 0
            else:
                logging.error(f"Failed to save {save_path}")
                return False, 0, 0
            
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False, 0, 0

# Process each image
for idx, row in tqdm(images_df.iterrows(), total=len(images_df)):
    if row['iconclass'] in iconclass_counts.index:
        filename = row['painting'].split('/')[-1] + '.jpg'
        save_path = os.path.join(jpeg_images_dir, filename)
        
        # Use the enhanced download function
        success, width, height = download_image(
            row['image'], 
            save_path,
            max_pixels=178956970,
            target_size=None  # Set to None to preserve original size
        )
        
        if success:
            # Store the image dimensions
            image_sizes.append((width, height))
            
            # Store the image and its class in the list
            image_data.append({
                'painting': row['painting'],
                'image': row['image'],
                'iconclass': row['iconclass'],
                'width': width,
                'height': height
            })
        else:
            # Track failed downloads
            failed_downloads.append({
                'painting': row['painting'],
                'image': row['image'],
                'iconclass': row['iconclass']
            })
        
        # Save the data to a JSON file every 50 images
        if (idx + 1) % 50 == 0:
            with open(os.path.join(wikidata_dir, 'wikidata_original.json'), 'w') as f:
                json.dump(image_data, f)
            logging.info(f"Saved data for {len(image_data)} images to JSON")

# Save any remaining data to the JSON file
with open(os.path.join(wikidata_dir, 'wikidata_original.json'), 'w') as f:
    json.dump(image_data, f)

# Save failed downloads to a separate file
with open(os.path.join(wikidata_dir, 'failed_downloads.json'), 'w') as f:
    json.dump(failed_downloads, f)
logging.info(f"Saved {len(failed_downloads)} failed downloads to JSON")

# Calculate statistics on image dimensions
if image_sizes:
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]
    
    width_mean = np.mean(widths)
    width_std = np.std(widths)
    height_mean = np.mean(heights)
    height_std = np.std(heights)
    
    print(f"Image Statistics:")
    print(f"Width: {width_mean:.2f} ± {width_std:.2f} pixels")
    print(f"Height: {height_mean:.2f} ± {height_std:.2f} pixels")
    print(f"Target from paper: Width: 778.84 ± 198.74, Height: 669.36 ± 174.18")
    print(f"\nTotal images: {len(image_sizes)}")
    print(f"Min width: {min(widths)}, Max width: {max(widths)}")
    print(f"Min height: {min(heights)}, Max height: {max(heights)}")
    print(f"Failed downloads: {len(failed_downloads)}")
    
    # Save statistics to a file
    with open(os.path.join(wikidata_data_dir, 'image_statistics.txt'), 'w') as f:
        f.write(f"Image Statistics:\n")
        f.write(f"Width: {width_mean:.2f} ± {width_std:.2f} pixels\n")
        f.write(f"Height: {height_mean:.2f} ± {height_std:.2f} pixels\n")
        f.write(f"Target from paper: Width: 778.84 ± 198.74, Height: 669.36 ± 174.18\n")
        f.write(f"\nTotal images: {len(image_sizes)}\n")
        f.write(f"Min width: {min(widths)}, Max width: {max(widths)}\n")
        f.write(f"Min height: {min(heights)}, Max height: {max(heights)}\n")
        f.write(f"Failed downloads: {len(failed_downloads)}\n")
    
    logging.info("Saved image statistics")

print("Image download complete.")

# Create test and ground truth files
test_images = []
ground_truth = []

# Iterate over each object in the image data
for item in image_data:
    # Extract the image filename
    image_filename = item['painting'].replace('http://www.wikidata.org/entity/', '')
    image_path = os.path.join(jpeg_images_dir, f'{image_filename}.jpg')
    
    # Check if the image exists in JPEGImages directory
    if os.path.exists(image_path):
        # Add the image filename to the test file list
        test_images.append(image_filename)
        
        # Add the object to the ground truth list
        ground_truth.append({
            'item': image_filename,
            'class': item['iconclass'],
            'width': item.get('width', 0),
            'height': item.get('height', 0)
        })

# Write the test images to 2_test_original.txt
with open(os.path.join(wikidata_data_dir, '2_test_original.txt'), 'w') as f:
    for image in test_images:
        f.write(f"{image}\n")

# Write the ground truth data to 2_ground_truth_original.json
with open(os.path.join(wikidata_data_dir, '2_ground_truth_original.json'), 'w') as f:
    json.dump(ground_truth, f)

print(f"Files 2_test_original.txt and 2_ground_truth_original.json have been created.")
print(f"Downloaded {len(test_images)} images with original sizes and aspect ratios.")
print(f"Failed downloads: {len(failed_downloads)}")
logging.info(f"Process complete. Downloaded {len(test_images)} images, failed {len(failed_downloads)}")
