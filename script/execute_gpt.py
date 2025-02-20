import os
import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import openai
import base64
import json
from configparser import ConfigParser

# Load OpenAI API key from environment variable
config = ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'gpt_data', 'psw.ini'))

OPENAI_API_KEY = config.get('openai', 'api_key', fallback=None)

if not OPENAI_API_KEY:
  raise ValueError("OpenAI API key is not set in the config file.")

def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as file:
            return json.load(file)
    return {}

def save_cache(cache, cache_file):
    with open(cache_file, 'w') as file:
        json.dump(cache, file, indent=4)

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

def classify_images_gpt(images, model, classes, system_prompt, test, limit=-1, batch_size=10):
    all_probs = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    client = openai.Client(api_key=OPENAI_API_KEY)
    cache_file = os.path.join(os.path.dirname(__file__), 'gpt_data', f'cache_{model}_{test}.json')
    cache = load_cache(cache_file)
    
    print(f"Using model: {model}")

    if limit > 0:
      images = images[:limit]
    
    few_shot_messages = []
    if test in ['test_3', 'test_4']:
      few_shot_file = os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', 'few-shot', 'train_data.csv')
      few_shot_df = pd.read_csv(few_shot_file)
      for _, row in few_shot_df.iterrows():
        image_path = os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', 'few-shot', f'{row["item"]}.jpg')
        few_shot_messages.append(
          {"role": "user", "content": [{"type": "image_url", "image_url": {"url": encode_image(image_path)}}]},
        )
        few_shot_messages.append(
          {"role": "assistant", "content": row['class']}
        )
    
    for i in tqdm(range(0, len(images), batch_size), desc="Processing Images", unit="batch"):
        batch = images[i:i+batch_size]
        image_urls = []
        batch_items = []
        
        for item, image_path in batch:
            if item in cache:
                all_probs.append(cache[item])
            else:
                image_urls.append({"type": "image_url", "image_url": {"url": encode_image(image_path)}})
                batch_items.append(item)
        
        if not image_urls:
            continue 
        
        try:
          messages = [{"role": "system", "content": system_prompt}] + few_shot_messages + [{"role": "user", "content": image_urls}]
          response = client.chat.completions.create(
            model=model,
            messages=messages
          )
            
          response_texts = []
          for choice in response.choices:
            content = choice.message.content
            try:
              json_content = json.loads(content)
              response_texts = json_content
            except json.JSONDecodeError:
              response_texts.append(content.split('\n'))
          input_tokens = response.usage.prompt_tokens
          output_tokens = response.usage.completion_tokens
          total_input_tokens += input_tokens
          total_output_tokens += output_tokens
          
          response_texts = list(response_texts.values())
            
          if len(response_texts) == len(batch_items):
              for idx, item in enumerate(batch_items):
                  probabilities = np.zeros(len(classes))
                  for cls_idx, (cls_id, _) in enumerate(classes):
                      if response_texts[idx] == cls_id:
                          probabilities[cls_idx] = 1.0
                  all_probs.append(probabilities)
                  cache[item] = probabilities.tolist()
          else:
              print(f"Warning: Mismatch between response texts and batch items. Skipping batch.")
              for _ in batch_items:
                  all_probs.append(np.zeros(len(classes)))
            
        except Exception as e:
            print(f"Error processing batch: {e}")
            for _ in batch_items:
                all_probs.append(np.zeros(len(classes)))
    
    save_cache(cache, cache_file)
    
    # Estimated cost calculation
    if model == "gpt-4o":
      cost_per_1m_input_tokens = 2.5  # Cost per 1M input tokens
      cost_per_1m_output_tokens = 10  # Cost per 1M output tokens

    if model == "gpt-4o-mini":
      cost_per_1m_input_tokens = 0.150  # Cost per 1M input tokens for gpt-4o-mini
      cost_per_1m_output_tokens = 0.600  # Cost per 1M output tokens for gpt-4o-mini
    
    total_cost = ((total_input_tokens / 1_000_000) * cost_per_1m_input_tokens) + ((total_output_tokens / 1_000_000) * cost_per_1m_output_tokens)
    
    print(f"Total input tokens used: {total_input_tokens}")
    print(f"Total output tokens used: {total_output_tokens}")
    print(f"Total cost per {len(all_probs)} images: ${total_cost:.4f}")

    estimated_cost_per_1864_images = total_cost * (1864 / len(images))
    print(f"Estimated cost for 1864 images: ${estimated_cost_per_1864_images:.4f}")
    
    return np.array(all_probs)

@click.command()
@click.option('--folders', multiple=True, default=['test_1', 'test_2'], help='List of folders to use')
@click.option('--models', multiple=True, help='List of model names to use')
@click.option('--limit', default=-1, help='Limit the number of images to process')
@click.option('--batch_size', default=10, help='Number of images per batch')
def main(folders, models, limit, batch_size):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    with open(os.path.join(dataset_dir, '2_test.txt'), 'r') as file:
        test_items = file.read().splitlines()
    
    images = load_images(test_items, os.path.join(dataset_dir, 'ArtDL', 'JPEGImages'))
    
    print(f"Number of images: {len(images)}\n")
    
    for folder in folders:
      for model in models:
        classes_df = pd.read_csv(os.path.join(dataset_dir, 'classes.csv'))
        
        system_prompt_name = 'system_prompt.txt'
        
        if folder in ['test_1', 'test_3']:
          classes = list(zip(classes_df['ID'], classes_df['Label']))
        elif folder in ['test_2', 'test_4']:
          classes = list(zip(classes_df['ID'], classes_df['Description']))
          system_prompt_name = 'system_prompt_description.txt'

        with open(os.path.join(os.path.dirname(__file__), 'gpt_data', system_prompt_name), 'r') as file:
          system_prompt = file.read()
        
        print(f"Processing images for test: {folder}")
        all_probs = classify_images_gpt(images, model, classes, system_prompt, folder, limit, batch_size)
        
        output_folder = os.path.join(base_dir, folder, model)
        os.makedirs(output_folder, exist_ok=True)
        np.save(os.path.join(output_folder, 'probs.npy'), all_probs)
        print(f"Probabilities shape: {all_probs.shape}\n")

if __name__ == '__main__':
    main()
