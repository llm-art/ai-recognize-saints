import os
import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import openai
import base64
from configparser import ConfigParser

# Load OpenAI API key from environment variable
config = ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'psw.ini'))

OPENAI_API_KEY = config.get('openai', 'api_key', fallback=None)

if not OPENAI_API_KEY:
  raise ValueError("OpenAI API key is not set in the config file.")

def load_images(test_items, dataset_dir):
    images = []
    for item in test_items:
        image_path = os.path.join(dataset_dir, f"{item}.jpg")
        if os.path.exists(image_path):
            images.append((item, image_path))
        else:
            print(f"Warning: Image {image_path} not found.")
    return images

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{image_base64}"

def classify_images_gpt(images, model, classes, limit=-1):
  all_probs = []

  client = openai.Client(api_key=OPENAI_API_KEY)
  
  print(f"Using model: {model}")
  
  for idx, (item, image_path) in enumerate(tqdm(images, desc="Processing Images", unit="image")):
    if limit > -1 and idx >= limit:
      break
    try:
      classes_str = 'Possible classes: \n'
      for cls in classes:
        classes_str += f'{cls[0]}, ({cls[1]})\n'
      image_base64_url = encode_image(image_path)
      response = client.chat.completions.create(
        model=model,
        messages=[
          {"role": "system", "content": "Classify the given image into one of the provided categories. You must choose one category and provide only the class name as the output."},
          {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_base64_url}},
            {"type": "text", "text": classes_str}
          ]}
        ]
      )
      
      response_text = response.choices[0].message.content
      print(f"Image {item}: {response_text}")
      probabilities = np.zeros(len(classes))
      
      for idx, (i, _) in enumerate(classes):
        if i.lower() in response_text.lower():
          probabilities[idx] = 1.0  # Simulated one-hot encoding
      
      all_probs.append(probabilities)
    except Exception as e:
      print(f"Error processing image {item}: {e}")
      all_probs.append(np.zeros(len(classes)))
  
  return np.array(all_probs)

@click.command()
@click.option('--folders', multiple=True, default=['test_1', 'test_2'], help='List of folders to use')
@click.option('--models', multiple=True, help='List of model names to use')
@click.option('--limit', default=-1, help='Limit the number of images to process')
def main(folders, models, limit):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    with open(os.path.join(dataset_dir, '2_test.txt'), 'r') as file:
        test_items = file.read().splitlines()
    
    images = load_images(test_items, os.path.join(dataset_dir, 'ArtDL', 'JPEGImages'))
    
    print(f"Number of images: {len(images)}\n")
    
    for folder in folders:
      for model in models:
        classes_df = pd.read_csv(os.path.join(dataset_dir, 'classes.csv'))
        
        if folder == 'test_1':
            classes = list(zip(classes_df['ID'], classes_df['Label']))
        else:
            classes = list(zip(classes_df['ID'], classes_df['Description']))
        
        print(f"Processing images for test: {folder}")
        all_probs = classify_images_gpt(images, model, classes, limit)
        
        output_folder = os.path.join(base_dir, folder, model)
        os.makedirs(output_folder, exist_ok=True)
        np.save(os.path.join(output_folder, 'probs.npy'), all_probs)
        print(f"Probabilities shape: {all_probs.shape}\n")

if __name__ == '__main__':
    main()