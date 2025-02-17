import os
import torch
import clip
import click
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

list_image_path = [
    "0c8573aa-ad2d-4672-bc2c-7067bd863153_bb1e7952-4766-41b9-bfdf-1abf01bac531.jpg",
    "2e9faf04-90cf-4973-b253-c77c53dd1ccf_f450ccb9-2973-442a-89a4-fa54eeeedd20.jpg",
    "1942_9_17_c.jpg",
    "1440147397.jpg",
    "1828079898.jpg"
]

list_txt = [
    "11H(JEROME)",
    "11H(DOMINIC)",
    "11H(FRANCIS)",
    "11H(PETER)",
    "11H(PAUL)"
]

def convert_models_to_fp32(model): 
    """
    Convert all model parameters and their gradients to float32.
    Useful if you're toggling between half precision and float 
    to avoid numerical issues.
    """
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float() 

class CustomImageDataset(Dataset):
    """
    Loads each image and text label. Applies the provided 'transform'
    for data augmentation + normalization, and uses clip.tokenize
    to convert text to tokens.
    """
    def __init__(self, list_image_path, list_txt, folder_images, transform):
        self.image_paths = [os.path.join(folder_images, img_path) for img_path in list_image_path]
        self.texts = clip.tokenize(list_txt)
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  # Apply augmentations + normalization
        text = self.texts[idx]
        return image, text

@click.command()
@click.option('--models', multiple=True, 
              default=['clip-vit-base-patch32', 'clip-vit-base-patch16', 'clip-vit-large-patch14'], 
              help='List of model names to use')
@click.option('--models_clip', multiple=True, 
              default=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], 
              help='Corresponding CLIP architectures to use')
@click.option('--num_epochs', default=150, help='Number of epochs to train')
def main(models, models_clip, num_epochs):

    augment_transforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
      )
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name, model_clip in zip(models, models_clip):

        model, _ = clip.load(model_clip, device=device, jit=False)
        model = model.to(device)

        # Freeze all parameters, unfreeze only final layers
        for param in model.parameters():
            param.requires_grad = False

        if hasattr(model.visual, 'proj') and model.visual.proj is not None:
            model.visual.proj.requires_grad = True
        model.text_projection.requires_grad = True

        # Prepare dataset & loader
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        image_folder = os.path.join(curr_dir, os.pardir, "dataset", "ArtDL", "JPEGImages/")

        list_labels = list_txt

        dataset = CustomImageDataset(list_image_path, list_labels, image_folder, augment_transforms)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)

        # Define optimizer & losses
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=1e-6)
        loss_img = torch.nn.CrossEntropyLoss()
        loss_txt = torch.nn.CrossEntropyLoss()

        print(f"\nTraining {model_name} ({model_clip}) for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}/{num_epochs}")
            for batch in pbar:
                optimizer.zero_grad()

                images, texts = batch
                images = images.to(device)
                texts = texts.to(device)

                # Forward pass
                logits_per_image, logits_per_text = model(images, texts)

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
                else: 
                    convert_models_to_fp32(model)
                    optimizer.step()
                    clip.model.convert_weights(model)

                pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {total_loss.item():.4f}")

        # Save the fine-tuned model
        output_folder = os.path.join(curr_dir, os.pardir, 'test_3', model_name)
        os.makedirs(output_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_folder, "clip.pth"))
        print(f"Model saved to {os.path.join(output_folder, 'clip.pth')}")

if __name__ == '__main__':
    main()