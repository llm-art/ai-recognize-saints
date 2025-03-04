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
            try:
                # Remove entries with arrays of only 0s from the cache
                cache = json.load(file)
                valid_cache = {k: v for k, v in cache.items() 
                              if isinstance(v, list) and len(v) > 0 and not all(x == 0 for x in v)}
                print(f"Loaded {len(valid_cache)} valid cached results")
                return valid_cache
            except json.JSONDecodeError:
                print(f"Warning: Cache file {cache_file} is corrupted. Creating new cache.")
                return {}
    return {}

def save_cache(cache, cache_file):
    with open(cache_file, 'w') as file:
        json.dump(cache, file, indent=4)

def save_cache_periodic(cache, cache_file, batch_count, save_frequency=5):
    """Periodically save cache to avoid losing results on crash"""
    if batch_count % save_frequency == 0:
        with open(cache_file, 'w') as file:
            json.dump(cache, file, indent=4)
        print(f"Cache saved after {batch_count} batches")

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

def classify_images_gpt(images, model, classes, system_prompt, test, dataset, limit=-1, batch_size=10, save_frequency=5):
    all_probs = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    client = openai.Client(api_key=OPENAI_API_KEY)
    cache_file = os.path.join(os.path.dirname(__file__), 'gpt_data', f'cache_{model}_{dataset}_{test}.json')
    cache = load_cache(cache_file)
    
    print(f"Using model: {model}")

    if limit > 0:
      images = images[:limit]
    
    few_shot_messages = []
    if test in ['test_3', 'test_4']:
      few_shot_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'dataset', f"{dataset}-data", 'few-shot')
      few_shot_file = os.path.join(few_shot_folder, 'train_data.csv')
      few_shot_df = pd.read_csv(few_shot_file)
      
      # Get class descriptions for better few-shot examples
      class_descriptions = {}
      for cls_id, cls_desc in classes:
          class_descriptions[cls_id] = cls_desc
      
      for _, row in few_shot_df.iterrows():
        image_path = os.path.join(few_shot_folder, f'{row["item"]}.jpg')
        
        # Enhanced user message with clear instruction
        few_shot_messages.append(
          {"role": "user", "content": [
            {"type": "text", "text": "Please classify this image into one of the provided categories."},
            {"type": "image_url", "image_url": {"url": encode_image(image_path)}}
          ]},
        )
        
        # Enhanced assistant response with reasoning
        class_id = row['class']
        class_desc = class_descriptions.get(class_id, "")
        
        # Create a more detailed response with reasoning
        assistant_response = f"This image depicts {class_id}"
        if class_desc:
            assistant_response += f" ({class_desc})"
        assistant_response += "."
        
        few_shot_messages.append(
          {"role": "assistant", "content": assistant_response}
        )
    
    batch_count = 0
    for i in tqdm(range(0, len(images), batch_size), desc="Processing Images", unit="batch"):
        batch = images[i:i+batch_size]
        content = [
            {"type": "text", "text": "Please classify the following set of images:"}
        ]
        batch_items = []
        
        for item, image_path in batch:
            if item in cache:
                all_probs.append(cache[item])
            else:
                content.append({"type": "text", "text": f"Image (ID: {item}):"})
                content.append({"type": "image_url", "image_url": {"url": encode_image(image_path)}})
                batch_items.append(item)
        
        if not batch_items:
            continue
        
        batch_count += 1
        try:
          messages = [{"role": "system", "content": system_prompt}] + few_shot_messages + [{"role": "user", "content": content}]
          response = client.chat.completions.create(
            model=model,
            messages=messages
          )
            
          # Extract response content
          content = response.choices[0].message.content
          input_tokens = response.usage.prompt_tokens
          output_tokens = response.usage.completion_tokens
          total_input_tokens += input_tokens
          total_output_tokens += output_tokens
          
          # Parse response with better error handling
          try:
              # Try to parse as JSON
              json_content = json.loads(content)
              
              # Check if we have the expected structure
              if isinstance(json_content, dict):
                  response_dict = {}
                  
                  # Handle different possible JSON structures
                  if all(key.startswith("image_") for key in json_content.keys()):
                      # Format: {"image_1": "CLASS_ID", ...}
                      for i, item in enumerate(batch_items):
                          image_key = f"image_{i+1}"
                          if image_key in json_content:
                              response_dict[item] = json_content[image_key]
                  else:
                      # Direct mapping or other format
                      for item in batch_items:
                          if item in json_content:
                              response_dict[item] = json_content[item]
                          elif str(item) in json_content:
                              response_dict[item] = json_content[str(item)]
                  
                  # Convert to list format expected by downstream code
                  response_texts = list(response_dict.values())
              else:
                  # Handle unexpected JSON structure (like array)
                  response_texts = json_content if isinstance(json_content, list) else [json_content]
          except json.JSONDecodeError:
              # Fall back to text parsing if JSON fails
              print(f"Warning: Failed to parse JSON response. Attempting text parsing.")
              lines = content.strip().split('\n')
              response_texts = []
              
              for line in lines:
                  # Try to extract class IDs from text
                  for cls_id, _ in classes:
                      if cls_id in line:
                          response_texts.append(cls_id)
                          break
          
          # Process the parsed responses
          if len(response_texts) == len(batch_items):
              for idx, item in enumerate(batch_items):
                  probabilities = np.zeros(len(classes))
                  for cls_idx, (cls_id, _) in enumerate(classes):
                      if isinstance(response_texts[idx], str) and response_texts[idx] == cls_id:
                          probabilities[cls_idx] = 1.0
                  all_probs.append(probabilities)
                  cache[item] = probabilities.tolist()
          else:
              print(f"Warning: Mismatch between response texts ({len(response_texts)}) and batch items ({len(batch_items)}). Skipping batch.")
              print(f"Response: {response_texts}")
              for _ in batch_items:
                  all_probs.append(np.zeros(len(classes)))
          
          # Periodically save cache
          save_cache_periodic(cache, cache_file, batch_count, save_frequency)
            
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
@click.option('--save_frequency', default=5, help='How often to save cache (in batches)')
@click.option('--datasets', multiple=True, default=['ArtDL'], help='List of datasets to use')
def main(folders, models, limit, batch_size, save_frequency, datasets):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    
    for dataset in datasets:
      
      dataset_dir = os.path.join(base_dir, 'dataset', dataset)
      dataset_data_dir = os.path.join(base_dir, 'dataset', f'{dataset}-data')
      
      with open(os.path.join(dataset_data_dir, '2_test.txt'), 'r') as file:
          test_items = file.read().splitlines()
      
      images = load_images(test_items, os.path.join(dataset_dir, 'JPEGImages'))
      
      print(f"Number of images: {len(images)}\n")

      print(f"Processing dataset: {dataset}")
      
      for folder in folders:
        for model in models:
          classes_df = pd.read_csv(os.path.join(dataset_data_dir, 'classes.csv'))
          
          # Determine the base system prompt name
          system_prompt_base = f'system_prompt_{dataset.lower()}'

          if folder in ['test_1', 'test_3']:
            classes = list(zip(classes_df['ID'], classes_df['Label']))
          elif folder in ['test_2', 'test_4']:
            classes = list(zip(classes_df['ID'], classes_df['Description']))
            system_prompt_base += '_description'
          
          # Try to use enhanced system prompt first, fall back to original if not found
          system_prompt_enhanced = f"{system_prompt_base}_enhanced.txt"
          system_prompt_original = f"{system_prompt_base}.txt"
          
          system_prompt_path = os.path.join(os.path.dirname(__file__), 'gpt_data', system_prompt_enhanced)
          if not os.path.exists(system_prompt_path):
            system_prompt_path = os.path.join(os.path.dirname(__file__), 'gpt_data', system_prompt_original)
            print(f"Using original system prompt: {system_prompt_original}")
          else:
            print(f"Using enhanced system prompt: {system_prompt_enhanced}")
          
          with open(system_prompt_path, 'r') as file:
            system_prompt = file.read()
          
          print(f"Processing images for test: {folder}")
          all_probs = classify_images_gpt(images, model, classes, system_prompt, folder, dataset, limit, batch_size, save_frequency)
          
          output_folder = os.path.join(base_dir, folder, dataset, model)
          os.makedirs(output_folder, exist_ok=True)
          np.save(os.path.join(output_folder, 'probs.npy'), all_probs)
          print(f"Probabilities shape: {all_probs.shape}\n")

if __name__ == '__main__':
    main()
