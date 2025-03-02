import torch
import clip
import pandas as pd
import os
from PIL import Image
from torchvision import transforms

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define model architecture (make sure it matches your fine-tuned model)
model_name = "ViT-B/32"  # Adjust based on your fine-tuned model
curr_dir = os.path.dirname(os.path.abspath(__file__))
fine_tuned_model_path = os.path.join(curr_dir, os.pardir, os.pardir, "test_3", "clip-vit-base-patch32", "clip.pth")

# Load fine-tuned CLIP model
model, _ = clip.load(model_name, device=device, jit=False)
model.load_state_dict(torch.load(fine_tuned_model_path, map_location=device, weights_only=False))
model.eval()  # Set to evaluation mode

base_model, _ = clip.load("ViT-B/32", device=device, jit=False)
base_model.eval()

# Define transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    )
])

# Load ground truth data
csv_path = os.path.join(curr_dir, 'train_data.csv')
df = pd.read_csv(csv_path)
df.rename(columns={"item": "image_path", "class": "text_label"}, inplace=True)

# Ensure CSV has correct columns
if "image_path" not in df.columns or "text_label" not in df.columns:
    raise ValueError("CSV file must contain 'image_path' and 'text_label' columns")

# Prepare text labels for classification
text_labels = df["text_label"].unique().tolist()
text_tokens = clip.tokenize(text_labels).to(device)

# Run inference
results = []

for _, row in df.iterrows():
  image_name = row["image_path"]
  image_path = os.path.join(curr_dir, f'{image_name}.jpg')
  true_label = row["text_label"]

  # Load and preprocess image
  image = Image.open(image_path).convert("RGB")
  image = transform(image).unsqueeze(0).to(device)

  # Compute similarity
  with torch.no_grad():
      image_features = model.encode_image(image)
      text_features = model.encode_text(text_tokens)

      # Normalize features
      image_features /= image_features.norm(dim=-1, keepdim=True)
      text_features /= text_features.norm(dim=-1, keepdim=True)

      # Compute cosine similarity
      similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
      predicted_index = similarity.argmax().item()
      predicted_label = text_labels[predicted_index]

  with torch.no_grad():
    base_image_features = base_model.encode_image(image)
    base_text_features = base_model.encode_text(text_tokens)

    base_image_features /= base_image_features.norm(dim=-1, keepdim=True)
    base_text_features /= base_text_features.norm(dim=-1, keepdim=True)

    base_similarity = (100.0 * base_image_features @ base_text_features.T).softmax(dim=-1)
    base_pred_label = text_labels[base_similarity.argmax().item()]

  results.append({
      "IMG": image_name,
      "GT": true_label,
      "CLIP base": base_pred_label,
      "CLIP fine-tuned": predicted_label
  })

# Convert results to DataFrame and save
results_df = pd.DataFrame(results)
output_csv_path = os.path.join(curr_dir, "few-shot_test.csv")
results_df.to_csv(output_csv_path, index=False)

print(f"Classification completed! Results saved to {output_csv_path}")