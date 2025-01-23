import os
import torch
import click
import numpy as np
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
@click.option('--folders', multiple=True, default=['test_1', 'test_2'], help='List of folders to use')
def main(models, folders):
    
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dataset_dir = os.path.join(base_dir, 'dataset')

    # Open test.txt and read the lines
    with open(os.path.join(dataset_dir, '2_test.txt'), 'r') as file:
        test_items = file.read().splitlines()

    images = load_images(test_items, os.path.join(dataset_dir, 'ArtDL', 'JPEGImages'))

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "true"

    print(f"Number of images: {len(images)}\n")

    # Check if a GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    for folder in folders:
      for model_name in models:

        # Load the model and processor
        processor = AutoProcessor.from_pretrained(f'openai/{model_name}')
        model = AutoModelForZeroShotImageClassification.from_pretrained(f'openai/{model_name}').to(device)

        # Load classes
        classes_df = pd.read_csv(os.path.join(dataset_dir, 'classes.csv'))
        
        if folder == 'test_1':
          classes = list(zip(classes_df['ID'], classes_df['Label']))
        else:
          classes = list(zip(classes_df['ID'], classes_df['Description']))

        # Break images into smaller batches
        batch_size = 16
        images_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

        all_probs = []
        print("#####################################################")
        print(f"# Processing images for test: {folder}")
        print(f"# Model: {model_name}")
        with tqdm(total=len(images), desc="# Processing Images", unit="image") as pbar:
          for batch_index, batch in enumerate(images_batches):
            try:
              # Process the batch
              inputs = processor(text=[cls[1] for cls in classes], images=batch, return_tensors="pt", padding=True).to(device)
              outputs = model(**inputs)

              # Get probabilities for the batch
              logits_per_image = outputs.logits_per_image  
              batch_probs = logits_per_image.softmax(dim=1)
              all_probs.append(batch_probs.detach().cpu().numpy())

              pbar.update(len(batch))
            except Exception as e:
              print(f"Error processing batch {batch_index + 1}: {e}")
              pbar.update(len(batch))

        # Get one tensor with all the probabilities
        all_probs = np.concatenate(all_probs, axis=0)
        print(f"Probabilities shape: {all_probs.shape}\n")

        # Convert all_probs to a DataFrame and store it as a CSV file
        output_folder = os.path.join(base_dir, folder, model_name)
        os.makedirs(output_folder, exist_ok=True)
        np.save(os.path.join(output_folder, 'probs.npy'), all_probs)

if __name__ == '__main__':
    main()