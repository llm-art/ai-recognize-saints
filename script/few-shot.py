import os
import torch
import click
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Import CLIP
import clip
from transformers import AutoModel, AutoTokenizer 

# Model Mapping
arch_models = {
    # CLIP models
    'clip-vit-base-patch32': ('ViT-B/32', 'clip'),
    'clip-vit-base-patch16': ('ViT-B/16', 'clip'),
    'clip-vit-large-patch14': ('ViT-L/14', 'clip'),
    
    # SIGLIP models from Hugging Face
    'siglip-base-patch16-512': ('google/siglip-base-patch16-512', 'siglip'),
    'siglip-large-patch16-384': ('google/siglip-large-patch16-384', 'siglip'),
    'siglip-so400m-patch14-384': ('google/siglip-so400m-patch14-384', 'siglip')
}

list_image_path = [
    "0c8573aa-ad2d-4672-bc2c-7067bd863153_bb1e7952-4766-41b9-bfdf-1abf01bac531.jpg",
    "2e9faf04-90cf-4973-b253-c77c53dd1ccf_f450ccb9-2973-442a-89a4-fa54eeeedd20.jpg",
    "1942_9_17_c.jpg",
    "1440147397.jpg",
    "1828079898.jpg"
]

list_txt = [
    ("11H(JEROME)", "Jerome"),
    ("11H(DOMINIC)", "Saint Dominic"),
    ("11H(FRANCIS)", "Francis of Assisi"),
    ("11H(PETER)", "Peter"),
    ("11H(PAUL)", "Paul")
]

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
@click.option('--models', multiple=True, 
              default=['clip-vit-base-patch32', 'siglip-base-patch16-512'], 
              help='List of models to train (supports CLIP and SIGLIP)')
@click.option('--num_epochs', default=150, help='Number of epochs to train')
@click.option('--lr', default=1e-5, help='Learning rate')
def main(models, num_epochs, lr):

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        image_folder = os.path.join(curr_dir, os.pardir, "dataset", "ArtDL", "JPEGImages/")

        
        if "512" in model_name:
            image_size = 512
        elif "384" in model_name:
            image_size = 384
        else:
            image_size = 224  # Default for CLIP

        augment_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Ensure correct size for model
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

        dataset = CustomImageDataset(list_image_path, list_txt, image_folder, augment_transforms, tokenizer, model_type)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

        # Define optimizer & losses
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()

        output_folder = os.path.join(curr_dir, os.pardir, 'test_3', model_name)
        os.makedirs(output_folder, exist_ok=True)
        
        loss_data = []

        print(f"\n🚀 Training {model_name} ({model_arch}) for {num_epochs} epochs...")
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
                
                if device == "cpu":
                  optimizer.step()
                elif model_type == "clip": 
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