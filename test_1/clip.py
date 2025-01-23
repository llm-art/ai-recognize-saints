import os
import torch
import click
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

def load_images(test_items, dataset_dir):
  images = []
  for item in test_items:
    image_path = os.path.join(dataset_dir, f"{item}.jpg")
    try:
      image = Image.open(image_path)
      images.append(image)
    except Exception as e:
      print(f"Error loading image {image_path}: {e}")

  return images

@click.command()
@click.option('--models', multiple=True, help='List of model names to use')
def main(models):
  cur_dir = os.path.dirname(__file__)
  par_dir = os.path.join(cur_dir, os.pardir)

  # Open test.txt and read the lines
  with open(os.path.join(par_dir, '2_test.txt'), 'r') as file:
    test_items = file.read().splitlines()

  images = load_images(test_items, os.path.join(par_dir, 'ArtDL', 'JPEGImages'))

  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "true"

  print(f"Number of images: {len(images)}")
  
  for model_name in models:

    # Load the model and processor
    print(f'Loading model: {model_name}')
    processor = AutoProcessor.from_pretrained(f'openai/{model_name}')
    model = AutoModelForZeroShotImageClassification.from_pretrained(f'openai/{model_name}')

    # Load classes
    classes_df = pd.read_csv(os.path.join(par_dir, 'classes.csv'))
    classes = list(zip(classes_df['ID'], classes_df['Label']))

    # Break images into smaller batches
    batch_size = 16
    images_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

    all_probs = []
    with tqdm(total=len(images), desc="Processing Images", unit="image") as pbar:
      for batch_index, batch in enumerate(images_batches):
        try:
          # Process the batch
          inputs = processor(text=[cls[1] for cls in classes], images=batch, return_tensors="pt", padding=True)
          outputs = model(**inputs)
          
          # Get probabilities for the batch
          logits_per_image = outputs.logits_per_image  
          batch_probs = logits_per_image.softmax(dim=1)
          all_probs.append(batch_probs.detach())

          
          pbar.update(len(batch))
        except Exception as e:
          print(f"Error processing batch {batch_index + 1}: {e}")
          pbar.update(len(batch))

    # Get one tensor with all the probabilities
    all_probs = torch.cat(all_probs, dim=0)
    print(f"Probabilities shape: {all_probs.shape}")

    # Convert all_probs to a DataFrame and store it as a CSV file
    torch.save(all_probs, os.path.join(cur_dir, 'evaluations', model_name, 'probs.pt'))

if __name__ == '__main__':
  main()