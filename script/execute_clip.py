#!/usr/bin/env python3
import os
import torch
import click
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# For test_1 and test_2
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

# For test_3
import clip
import torch.nn as nn
import torch.nn.functional as F

def load_images(test_items, dataset_dir):
    """
    Loads images from disk given a list of item IDs and a directory.
    Returns a list of PIL Image objects.
    """
    images = []
    for item in test_items:
        image_path = os.path.join(dataset_dir, f"{item}.jpg")
        try:
            image = Image.open(image_path).convert("RGB")
            images.append((item, image))  # keep track of item name + image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
    return images

################################
# Example Linear Probe Definition
################################
# You must match the architecture
# you used for training the probe.
class LinearProbe(nn.Module):
    """
    Example linear probe: a single linear layer, taking CLIP embeddings -> class logits.
    Modify as needed if your actual saved classifier has a different structure.
    """
    def __init__(self, embed_dim=512, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x shape: [batch_size, embed_dim]
        return self.fc(x.float())

@click.command()
@click.option('--models', multiple=True, help='List of model names to use, e.g., clip-vit-base-patch32')
@click.option('--folders', multiple=True, default=['test_1', 'test_2', 'test_3', 'test_4'], help='List of folders to use')
def main(models, folders):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dataset_dir = os.path.join(base_dir, 'dataset')

    # 1) Load test items (image IDs) from file
    with open(os.path.join(dataset_dir, '2_test.txt'), 'r') as file:
        test_items = file.read().splitlines()

    # 2) Load actual PIL images
    images_data = load_images(test_items, os.path.join(dataset_dir, 'ArtDL', 'JPEGImages'))
    print(f"Number of images: {len(images_data)}\n")

    # 3) Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "true"

    # 4) For each folder and model, do inference
    for folder in folders:
        for model_name in models:
            output_folder = os.path.join(base_dir, folder, model_name)
            os.makedirs(output_folder, exist_ok=True)
            
            batch_size = 16

            # Read classes from CSV
            classes_df = pd.read_csv(os.path.join(dataset_dir, 'classes.csv'))
            
            if folder in ['test_1', 'test_3']:
                classes = list(zip(classes_df['ID'], classes_df['Label']))
            elif folder in ['test_2', 'test_4']:
                classes = list(zip(classes_df['ID'], classes_df['Description']))

            print("#####################################################")
            print(f"# Processing images for test: {folder}")
            print(f"# Model: {model_name}")

            if folder in ['test_1','test_2']:
                ##############################################
                # Zero-Shot with Hugging Face CLIP (HF style)
                ##############################################
                processor = AutoProcessor.from_pretrained(f'openai/{model_name}')
                hf_model = AutoModelForZeroShotImageClassification.from_pretrained(
                    f'openai/{model_name}'
                ).to(device)
                hf_model.eval()

                all_probs = []
                # Break images into smaller batches
                for i in tqdm(range(0, len(images_data), batch_size), desc="# Processing", unit="image"):
                    batch_items = images_data[i:i+batch_size]
                    batch_pil = [img for (_, img) in batch_items]

                    # Prepare text prompts from classes
                    text_prompts = [cls[1] for cls in classes]
                    inputs = processor(text=text_prompts, images=batch_pil, return_tensors="pt", padding=True).to(device)

                    with torch.no_grad():
                        outputs = hf_model(**inputs)
                        logits_per_image = outputs.logits
                        probs = logits_per_image.softmax(dim=1)  # shape [batch_size, num_classes]
                    all_probs.append(probs.cpu().numpy())

                all_probs = np.concatenate(all_probs, axis=0)  # shape [N, num_classes]
                print(f"Probabilities shape: {all_probs.shape}\n")
                np.save(os.path.join(output_folder, 'probs.npy'), all_probs)

            elif folder in ['test_3', 'test_4']:
                mapping = {
                  "clip-vit-base-patch32": "ViT-B/32",
                  "clip-vit-base-patch16": "ViT-B/16",
                  "clip-vit-large-patch14": "ViT-L/14"
                }
                
                fine_tuned_model_path = os.path.join(output_folder, 'model.pth')
                if not os.path.exists(fine_tuned_model_path):
                    raise FileNotFoundError(
                        f"Fine-tuned CLIP not found at {fine_tuned_model_path}!"
                    )
                
                actual_model = mapping.get(model_name)
                clip_model, preprocess = clip.load(actual_model, device=device, jit=False)
                clip_model.load_state_dict(torch.load(fine_tuned_model_path, map_location=device, weights_only=False))
                clip_model.eval()

                all_probs = []

                for i in tqdm(range(0, len(images_data), batch_size), desc="# Processing", unit="image"):
                  batch_items = images_data[i:i+batch_size]
                  batch_pil = [img for _, img in batch_items]
                  images_input = torch.stack([preprocess(img) for img in batch_pil]).to(device)

                  with torch.no_grad():
                    image_features = clip_model.encode_image(images_input)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    text_prompts = [c[1] for c in classes]
                    tokens = clip.tokenize(text_prompts).to(device)
                    text_features = clip_model.encode_text(tokens)
                    text_features /= text_features.norm(dim=-1, keepdim=True)

                    logits = (image_features @ text_features.T).softmax(dim=-1)
                    all_probs.append(logits.cpu().numpy())

                all_probs = np.concatenate(all_probs, axis=0)
                print(f"Probabilities shape: {all_probs.shape}\n")
                np.save(os.path.join(output_folder, 'probs.npy'), all_probs)

if __name__ == '__main__':
    main()