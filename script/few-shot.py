import os
import torch
import click
import csv
import pandas as pd
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Import CLIP
import clip
from transformers import AutoModel, AutoTokenizer 

CLIP_MODEL = 'clip'
SIGLIP_MODEL = 'siglip'

# Model Mapping
arch_models = {
  # CLIP models
  'clip-vit-base-patch32': ('ViT-B/32', CLIP_MODEL),
  'clip-vit-base-patch16': ('ViT-B/16', CLIP_MODEL),
  'clip-vit-large-patch14': ('ViT-L/14', CLIP_MODEL),
  
  # SIGLIP models from Hugging Face
  'siglip-base-patch16-512': ('google/siglip-base-patch16-512', SIGLIP_MODEL),
  'siglip-large-patch16-384': ('google/siglip-large-patch16-384', SIGLIP_MODEL),
  'siglip-so400m-patch14-384': ('google/siglip-so400m-patch14-384', SIGLIP_MODEL)
}

def convert_models_to_fp32(model):
    """Convert model parameters to float32"""
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() 

class CustomImageDataset(Dataset):
    """Loads images and text labels with preprocessing."""
    def __init__(self, list_image_path, list_txt, folder_images, transform, tokenizer, model_type):
        self.image_paths = [os.path.join(folder_images, img_path) for img_path in list_image_path]
        if model_type == "clip":
            self.texts = clip.tokenize([label[1] for label in list_txt])
        else:  # SIGLIP uses Hugging Face tokenizer
            self.texts = tokenizer([label[1] for label in list_txt], padding=True, return_tensors="pt")["input_ids"]
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  
        text = self.texts[idx]
        return image, text

@click.command()
@click.option('--folders', multiple=True, default=['test_3'], help='List of input folders')
@click.option('--models', multiple=True, 
              default=['clip-vit-base-patch32', 'siglip-base-patch16-512'], 
              help='List of models to train (supports CLIP and SIGLIP)')
@click.option('--num_epochs', default=150, help='Number of epochs to train')
@click.option('--lr', default=1e-5, help='Learning rate')
@click.option('--datasets', multiple=True, default=["ArtDL", "IconArt"], help='Number of epochs to train')
def main(folders, models, num_epochs, lr, datasets):

  for dataset_name in datasets:
    # Get the base directory for the current dataset
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(curr_dir, os.pardir, "dataset", f"{dataset_name}-data")
    
    # Read the classes.csv file to get class information
    classes_file = os.path.join(dataset_dir, "classes.csv")
    class_info = {}
    
    try:
      # Read the classes CSV file
      df_classes = pd.read_csv(classes_file)
      
      # Handle different column names in different datasets
      id_col = "ID" if "ID" in df_classes.columns else "ICONCLASS ID"
      label_col = "Label" if "Label" in df_classes.columns else "Label"
      
      # Create a mapping from class ID to label
      for _, row in df_classes.iterrows():
        class_info[row[id_col]] = row[label_col]
    except Exception as e:
      print(f"Error reading classes file for {dataset_name}: {e}")
      continue
    
    # Read the few-shot data to get image paths and classes
    # Try different possible filenames
    few_shot_dir = os.path.join(dataset_dir, "few-shot")
    possible_filenames = ["classes.csv", "train_data.csv", "few-shot_test.csv"]
    
    image_paths = []
    list_txt_base = []
    
    for filename in possible_filenames:
      few_shot_file = os.path.join(few_shot_dir, filename)
      if os.path.exists(few_shot_file):
        try:
          df_few_shot = pd.read_csv(few_shot_file)
          
          # Determine column names based on available columns
          img_col = None
          class_col = None
          
          if "IMG" in df_few_shot.columns:
            img_col = "IMG"
          elif "item" in df_few_shot.columns:
            img_col = "item"
            
          if "GT" in df_few_shot.columns:
            class_col = "GT"
          elif "class" in df_few_shot.columns:
            class_col = "class"
          
          if img_col and class_col:
            # Get image paths and class IDs
            for _, row in df_few_shot.iterrows():
              img_path = f"{row[img_col]}.jpg" if not str(row[img_col]).endswith(".jpg") else row[img_col]
              class_id = row[class_col]
              
              # Add to image paths
              image_paths.append(img_path)
              
              # Add to list_txt_base if the class ID exists in class_info
              if class_id in class_info:
                list_txt_base.append((class_id, class_info[class_id]))
              else:
                print(f"Warning: Class ID {class_id} not found in classes.csv for {dataset_name}")
            
            # Break after finding and processing the first valid file
            break
        except Exception as e:
          print(f"Error reading few-shot file {filename} for {dataset_name}: {e}")
    
    if not image_paths or not list_txt_base:
      print(f"No valid few-shot data found for {dataset_name}. Skipping.")
      continue
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for folder_name in folders:
      
      # Only test_3 is needed now (no descriptions)
      list_txt = list_txt_base
      
      for model_name in models:
          if model_name not in arch_models:
              print(f"Model {model_name} is not recognized. Skipping.")
              continue

          model_arch, model_type = arch_models[model_name]

          # Load model dynamically
          if model_type == 'clip':
              model, _ = clip.load(model_arch, device=device, jit=False)
              tokenizer = None  # CLIP has built-in tokenization
          elif model_type == 'siglip':
              model = AutoModel.from_pretrained(model_arch).to(device)
              tokenizer = AutoTokenizer.from_pretrained(model_arch)
          else:
              raise ValueError(f"Unknown model type: {model_type}")

          model = model.to(device)

          # Freeze all layers, unfreeze only final layers
          for param in model.parameters():
              param.requires_grad = False  

          # Unfreeze only the last transformer layers
          if model_type == "clip":
              for param in model.visual.transformer.resblocks[-1:].parameters():
                  param.requires_grad = True 
              for param in model.transformer.resblocks[-1:].parameters():
                  param.requires_grad = True 
          elif model_type == "siglip":
              for param in model.vision_model.encoder.layers[-1:].parameters():
                  param.requires_grad = True
              for param in model.text_model.encoder.layers[-1:].parameters():
                  param.requires_grad = True

          # Prepare dataset & loader
          curr_dir = os.path.dirname(os.path.abspath(__file__))
          image_folder = os.path.join(curr_dir, os.pardir, "dataset", f"{dataset_name}-data", "few-shot")

          
          if "512" in model_name:
              image_size = 512
          elif "384" in model_name:
              image_size = 384
          else:
              image_size = 224

          augment_transforms = transforms.Compose([
              transforms.Resize((image_size, image_size)),  # Ensure correct size for model
              transforms.ToTensor(),
              transforms.Normalize(
                  mean=(0.48145466, 0.4578275, 0.40821073),
                  std=(0.26862954, 0.26130258, 0.27577711)
              )
          ])

          dataset = CustomImageDataset(image_paths, list_txt, image_folder, augment_transforms, tokenizer, model_type)
          dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

          # Define optimizer & losses
          trainable_params = [p for p in model.parameters() if p.requires_grad]
          optimizer = torch.optim.Adam(trainable_params, lr=lr)
          loss_img = torch.nn.CrossEntropyLoss()
          loss_txt = torch.nn.CrossEntropyLoss()

          output_folder = os.path.join(curr_dir, os.pardir, folder_name, dataset_name, model_name)
          os.makedirs(output_folder, exist_ok=True)
          
          loss_data = []

          print(f"\nTraining {model_name} ({model_arch}) for {num_epochs} epochs...")
          for epoch in range(num_epochs):
              pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}/{num_epochs}")
              for batch in pbar:
                  optimizer.zero_grad()

                  images, texts = batch
                  images = images.to(device)
                  texts = texts.to(device)

                  # Forward pass
                  if model_type == "clip":
                      logits_per_image, logits_per_text = model(images, texts)
                  elif model_type == "siglip":
                      # Process images with SIGLIP's vision model
                      vision_outputs = model.vision_model(images)  # shape: (batch, num_patches, hidden_dim)
                      image_features = vision_outputs.pooler_output  # Global representation (batch, hidden_dim)
                      
                      # Process text with SIGLIP's text model
                      text_outputs = model.text_model(input_ids=texts).pooler_output  # shape: (batch, hidden_dim)
                      
                      # Normalize features
                      image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                      text_features = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
                      
                      # Compute similarity logits
                      logits_per_image = 100.0 * image_features @ text_features.T
                      logits_per_text = logits_per_image.T

                  # Compute loss
                  ground_truth = torch.arange(len(images), dtype=torch.long, device=device)
                  total_loss = (
                      loss_img(logits_per_image, ground_truth) +
                      loss_txt(logits_per_text, ground_truth)
                  ) / 2

                  # Backward pass
                  total_loss.backward()
                  
                  if device == "cpu" or model_type == "siglip":
                    optimizer.step()
                  else: 
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)

                  res = f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}"

                  pbar.set_description(res)
                  loss_data.append(res)

          # Save fine-tuned model and loss data
          with open(os.path.join(output_folder, "training_log.csv"), "w") as f:
            for line in loss_data:
              f.write(line + "\n")
          torch.save(model.state_dict(), os.path.join(output_folder, "model.pth"))
          print(f"Model saved to {os.path.join(output_folder, 'model.pth')}")

if __name__ == '__main__':
  main()
