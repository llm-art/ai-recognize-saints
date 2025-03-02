import os
import torch
import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, AutoModel

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
@click.option('--models', multiple=True, default=["siglip-base-patch16-512", "siglip-large-patch16-384", "siglip-so400m-patch14-384"], help='List of model names to use')
@click.option('--folders', multiple=True, default=['test_1', 'test_2', 'test_3'], help='List of folders to use')
@click.option('--datasets', multiple=True, default=['ArtDL', 'IconArt'], help='Name of the dataset directory')
def main(models, folders, datasets):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)

    
    for dataset in datasets:

      dataset_dir = os.path.join(base_dir, 'dataset', dataset)
      dataset_data_dir = os.path.join(base_dir, 'dataset', f'{dataset}-data')

      if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}!")

      # 1) Load test items (image IDs) from file
      with open(os.path.join(dataset_data_dir, '2_test.txt'), 'r') as file:
          test_items = file.read().splitlines()

      # 2) Load actual PIL images
      images_data = load_images(test_items, os.path.join(dataset_dir, 'JPEGImages'))
      print(f"Number of images: {len(images_data)}\n")

      # 3) Check device
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      print(f"Using device: {device}\n")

      os.environ["TOKENIZERS_PARALLELISM"] = "false"
      os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "true"

      for folder in folders:
        for model_name in models:
          output_folder = os.path.join(base_dir, folder, dataset, model_name)
          os.makedirs(output_folder, exist_ok=True)

          # Load the model and processor
          processor = AutoProcessor.from_pretrained(f'google/{model_name}')
          
          if folder in ["test_3", "test_4"]:
            model_path = os.path.join(base_dir, 'test_3', model_name, 'model.pth')
            model = AutoModel.from_pretrained(f'google/{model_name}').to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            print(f"Loaded fine-tuned model from {model_path}")
          else:
            # Load zero-shot SIGLIP model from Hugging Face
            model = AutoModelForZeroShotImageClassification.from_pretrained(f'google/{model_name}').to(device)

          # Load classes
          classes_df = pd.read_csv(os.path.join(dataset_data_dir, 'classes.csv'))
          
          if folder in ['test_1', 'test_3']:
            classes = list(zip(classes_df['ID'], classes_df['Label']))
          else:
            classes = list(zip(classes_df['ID'], classes_df['Description']))

          # Break images into smaller batches
          batch_size = 16
          images_batches = [images_data[i:i + batch_size] for i in range(0, len(images_data), batch_size)]

          all_probs = []
          print("#####################################################")
          print(f"# Processing images_data for test: {folder}")
          print(f"# Model: {model_name}")
          with tqdm(total=len(images_data), desc="# Processing images_data", unit="image") as pbar:
            for batch_index, batch in enumerate(images_batches):
              try:
                # Process the batch
                inputs = processor(text=[cls[1] for cls in classes], images=batch, padding="max_length", return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)

                logits_per_image = outputs.logits_per_image
                batch_probs = torch.sigmoid(logits_per_image)
                all_probs.append(batch_probs.detach().cpu().numpy())

                pbar.update(len(batch))
              except Exception as e:
                print(f"Error processing batch {batch_index + 1}: {e}")
                pbar.update(len(batch))

          # Get one tensor with all the probabilities
          all_probs = np.concatenate(all_probs, axis=0)
          print(f"Probabilities shape: {all_probs.shape}\n")

          # Convert all_probs to a numpy tensor and store it as a .npy file
          np.save(os.path.join(output_folder, 'probs.npy'), all_probs)

if __name__ == '__main__':
    main()