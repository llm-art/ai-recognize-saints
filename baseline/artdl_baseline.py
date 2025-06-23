import os
import json
import click
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Shared classes between ArtDL, ICONCLASS, and Wikidata
SHARED_CLASSES = [
    '11HH(MARY MAGDALENE)',  # Mary Magdalene
    '11H(JEROME)',           # Jerome/St. Jerome
    '11H(PETER)',            # Peter/St. Peter
    '11H(FRANCIS)',          # Francis/St. Francis
    '11H(SEBASTIAN)'         # Sebastian/St. Sebastian
]

class PadToSquare:
    """Pad image to square based on largest dimension (ArtDL methodology)"""
    def __call__(self, img):
        w, h = img.size
        max_dim = max(w, h)
        
        # Calculate padding
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        
        # Pad image
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        return transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

def load_pretrained_model(model_path, device, num_classes=10):
    """Load pre-trained model following the same pattern as execute_clip.py test_3"""
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # Create a standard ResNet50 model
    model = models.resnet50(pretrained=False)
    
    # Modify the final layer to match the number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Handle different state dict formats by removing prefixes if needed
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # Remove common prefixes that might exist
        clean_key = key
        for prefix in ['resnet50.', 'backbone.', 'model.', 'module.']:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
                break
        cleaned_state_dict[clean_key] = value
    
    # Try to load the cleaned state dict
    try:
        model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Warning: Could not load some weights: {e}")
        print("Proceeding with available weights...")
    
    return model

class EvaluationDataset(Dataset):
    """Dataset class for evaluation on ICONCLASS and Wikidata test sets"""
    def __init__(self, dataset_name, shared_classes_only=False, transform=None, image_size=224):
        self.dataset_name = dataset_name
        self.shared_classes_only = shared_classes_only
        self.transform = transform
        self.image_size = image_size
        
        # Set paths
        base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        self.data_dir = os.path.join(base_dir, 'dataset', f'{dataset_name}-data')
        self.images_dir = os.path.join(base_dir, 'dataset', dataset_name, 'JPEGImages')
        
        # Load ground truth and classes
        with open(os.path.join(self.data_dir, '2_ground_truth.json'), 'r') as f:
            ground_truth_list = json.load(f)
            self.ground_truth = {item['item']: item['class'] for item in ground_truth_list}
        
        # Load test set
        with open(os.path.join(self.data_dir, '2_test.txt'), 'r') as f:
            test_images = [line.strip() for line in f.readlines()]
        
        self.classes_df = pd.read_csv(os.path.join(self.data_dir, 'classes.csv'))
        
        # Filter classes based on modality
        if shared_classes_only:
            # Only keep shared classes
            self.classes_df = self.classes_df[self.classes_df['ID'].isin(SHARED_CLASSES)].reset_index(drop=True)
            valid_classes = set(SHARED_CLASSES)
            print(f"Using shared classes only: {len(valid_classes)} classes")
        else:
            # Use all classes
            valid_classes = set(self.classes_df['ID'].tolist())
            print(f"Using all classes: {len(valid_classes)} classes")
        
        # Create class mappings
        self.class_to_idx = {row['ID']: idx for idx, row in self.classes_df.iterrows()}
        self.idx_to_class = {idx: row['ID'] for idx, row in self.classes_df.iterrows()}
        
        # Filter test images to only include those with valid classes and existing ground truth
        self.image_list = [img for img in test_images 
                          if img in self.ground_truth and self.ground_truth[img] in valid_classes]
        
        print(f"Loaded {len(self.image_list)} test images for {dataset_name}")
        print(f"Classes: {sorted(valid_classes)}")
        
        # Filter out missing/corrupted images
        self._filter_valid_images()
        
    def __len__(self):
        return len(self.image_list)
    
    def _filter_valid_images(self):
        """Filter out images that cannot be loaded"""
        valid_images = []
        missing_count = 0
        
        print(f"Filtering valid images from {len(self.image_list)} total images...")
        
        for img_name in tqdm(self.image_list, desc="Checking image availability"):
            # Try different image extensions
            possible_extensions = ['.jpg', '.jpeg', '.png', '']
            image_found = False
            
            for ext in possible_extensions:
                img_path = os.path.join(self.images_dir, img_name + ext)
                if os.path.exists(img_path):
                    try:
                        # Try to open the image to ensure it's valid
                        with Image.open(img_path) as img:
                            img.verify()  # Verify the image is not corrupted
                        image_found = True
                        break
                    except Exception:
                        continue
            
            if image_found:
                valid_images.append(img_name)
            else:
                missing_count += 1
        
        print(f"Found {len(valid_images)} valid images, {missing_count} missing/corrupted images")
        self.image_list = valid_images
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        
        # Try different image extensions
        possible_extensions = ['.jpg', '.jpeg', '.png', '']
        image = None
        
        for ext in possible_extensions:
            img_path = os.path.join(self.images_dir, img_name + ext)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                    break
                except Exception:
                    continue
        
        # This should not happen after filtering, but keep as safety net
        if image is None:
            raise RuntimeError(f"Could not load image for {img_name} - this should not happen after filtering")
        
        # Get label
        class_id = self.ground_truth[img_name]
        label = self.class_to_idx[class_id]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default ArtDL preprocessing
            transform = transforms.Compose([
                PadToSquare(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        return image, label, img_name

def get_artdl_transforms(image_size=224):
    """Get transforms following ArtDL methodology for evaluation"""
    transform = transforms.Compose([
        PadToSquare(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform


def create_class_mapping(artdl_classes, target_classes):
    """Create mapping from ArtDL class indices to target dataset class indices"""
    # Load ArtDL classes
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    artdl_df = pd.read_csv(os.path.join(base_dir, 'dataset', 'ArtDL-data', 'classes.csv'))
    artdl_class_to_idx = {row['ID']: idx for idx, row in artdl_df.iterrows()}
    
    # Create mapping dictionary
    class_mapping = {}
    for target_idx, target_class in enumerate(target_classes):
        if target_class in artdl_class_to_idx:
            artdl_idx = artdl_class_to_idx[target_class]
            class_mapping[artdl_idx] = target_idx
        else:
            print(f"Warning: Target class {target_class} not found in ArtDL classes")
    
    return class_mapping

def map_predictions(artdl_predictions, class_mapping, num_target_classes):
    """Map ArtDL predictions to target dataset classes"""
    mapped_predictions = []
    
    for pred in artdl_predictions:
        if pred in class_mapping:
            mapped_predictions.append(class_mapping[pred])
        else:
            # If no mapping exists, predict the most frequent class (0)
            mapped_predictions.append(0)
    
    return np.array(mapped_predictions)

def evaluate_model(model, test_loader, dataset_name, save_dir, class_mapping, modality_name):
    """Evaluate model and save results"""
    model.eval()
    all_artdl_preds = []
    all_labels = []
    all_probs = []
    all_names = []
    
    with torch.no_grad():
        for images, labels, names in tqdm(test_loader, desc=f'Evaluating {modality_name}'):
            images = images.to(device)
            outputs = model(images)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_artdl_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_names.extend(names)
    
    # Map ArtDL predictions to target dataset classes
    num_target_classes = len(test_loader.dataset.classes_df)
    mapped_preds = map_predictions(all_artdl_preds, class_mapping, num_target_classes)
    
    # Map probabilities (only keep relevant classes)
    mapped_probs = np.zeros((len(all_probs), num_target_classes))
    for i, artdl_prob in enumerate(all_probs):
        for artdl_idx, target_idx in class_mapping.items():
            if artdl_idx < len(artdl_prob):
                mapped_probs[i, target_idx] = artdl_prob[artdl_idx]
    
    # Normalize probabilities
    row_sums = mapped_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    mapped_probs = mapped_probs / row_sums
    
    # Save predictions and probabilities
    np.save(os.path.join(save_dir, 'probs.npy'), mapped_probs)
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, mapped_preds) * 100
    
    # Create confusion matrix
    dataset = test_loader.dataset
    class_names = [dataset.classes_df.iloc[i]['ID'] for i in range(len(dataset.classes_df))]
    
    cm = confusion_matrix(all_labels, mapped_preds, labels=range(len(class_names)))
    
    # Save confusion matrix as CSV
    confusion_matrix_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    confusion_matrix_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted classes')
    plt.ylabel('Actual classes')
    plt.title(f'Confusion Matrix for {dataset_name} ({modality_name})')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate class metrics
    class_metrics(all_labels, mapped_preds, mapped_probs, dataset, save_dir, accuracy)
    
    return accuracy

def class_metrics(y_true_indices, y_pred_indices, probs, dataset, save_dir, accuracy):
    """Calculate and save class metrics"""
    
    # Constants
    KEY_CLASS_NAME = 'class_name'
    KEY_NUM_IMAGES = '# of Images'
    KEY_PRECISION = 'Precision'
    KEY_RECALL = 'Recall'
    KEY_F1_SCORE = 'F1 Score'
    KEY_AVG_PRECISION = 'Average Precision'
    
    classes = [(dataset.classes_df.iloc[i]['ID'], dataset.classes_df.iloc[i]['Label']) 
               for i in range(len(dataset.classes_df))]
    
    # Count images per class
    class_image_counts = {cls[0]: y_true_indices.count(i) for i, cls in enumerate(classes)}
    
    # One-hot encode for average precision calculation
    y_true_one_hot = label_binarize(y_true_indices, classes=range(len(classes)))
    if len(classes) == 2:
        # Handle binary classification case
        y_true_one_hot = np.column_stack([1 - y_true_one_hot, y_true_one_hot])
    
    # Class-level metrics
    class_precisions = precision_score(y_true_indices, y_pred_indices, average=None, 
                                     labels=range(len(classes)), zero_division=0) * 100
    class_recalls = recall_score(y_true_indices, y_pred_indices, average=None, 
                               labels=range(len(classes)), zero_division=0) * 100
    class_f1_scores = f1_score(y_true_indices, y_pred_indices, average=None, 
                             labels=range(len(classes)), zero_division=0) * 100
    
    # Handle average precision calculation
    try:
        class_avg_precisions = average_precision_score(y_true_one_hot, probs, average=None) * 100
    except:
        class_avg_precisions = np.zeros(len(classes)) * 100
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        KEY_CLASS_NAME: [cls[0] for cls in classes],
        KEY_NUM_IMAGES: [class_image_counts[cls[0]] for cls in classes],
        KEY_PRECISION: class_precisions,
        KEY_RECALL: class_recalls,
        KEY_F1_SCORE: class_f1_scores,
        KEY_AVG_PRECISION: class_avg_precisions
    })
    
    # Macro and Micro Averages
    mean_precision = precision_score(y_true_indices, y_pred_indices, average='macro', zero_division=0) * 100
    mean_recall = recall_score(y_true_indices, y_pred_indices, average='macro', zero_division=0) * 100
    mean_f1_score = f1_score(y_true_indices, y_pred_indices, average='macro', zero_division=0) * 100
    
    try:
        mean_avg_precision = average_precision_score(y_true_one_hot, probs, average='macro') * 100
    except:
        mean_avg_precision = 0.0
    
    micro_precision = precision_score(y_true_indices, y_pred_indices, average='micro', zero_division=0) * 100
    micro_recall = recall_score(y_true_indices, y_pred_indices, average='micro', zero_division=0) * 100
    micro_f1_score = f1_score(y_true_indices, y_pred_indices, average='micro', zero_division=0) * 100
    
    try:
        micro_avg_precision = average_precision_score(y_true_one_hot, probs, average='micro') * 100
    except:
        micro_avg_precision = 0.0
    
    # Append Macro and Micro metrics
    metrics_df = pd.concat([
        metrics_df,
        pd.DataFrame([{
            KEY_CLASS_NAME: 'Macro',
            KEY_NUM_IMAGES: '-',
            KEY_PRECISION: mean_precision,
            KEY_RECALL: mean_recall,
            KEY_F1_SCORE: mean_f1_score,
            KEY_AVG_PRECISION: mean_avg_precision
        }, {
            KEY_CLASS_NAME: 'Micro',
            KEY_NUM_IMAGES: '-',
            KEY_PRECISION: micro_precision,
            KEY_RECALL: micro_recall,
            KEY_F1_SCORE: micro_f1_score,
            KEY_AVG_PRECISION: micro_avg_precision
        }])
    ], ignore_index=True)
    
    # Format percentages
    for col in [KEY_PRECISION, KEY_RECALL, KEY_F1_SCORE, KEY_AVG_PRECISION]:
        metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else x)
    
    # Save as CSV
    metrics_df.to_csv(os.path.join(save_dir, 'class_metrics.csv'), index=False)
    
    # Save as Markdown
    metrics_df.to_markdown(os.path.join(save_dir, 'class_metrics.md'), index=False)
    
    # Create summary metrics
    summary_df = pd.DataFrame([{
        'Model': 'artdl-baseline',
        'Macro Average Precision': f"{mean_avg_precision:.2f}%",
        'Micro Average Precision': f"{micro_avg_precision:.2f}%",
        'Accuracy': f"{accuracy:.2f}%"
    }])
    
    summary_df.to_csv(os.path.join(save_dir, 'summary_metrics.csv'), index=False)
    
    return mean_avg_precision, micro_avg_precision

@click.command()
@click.option('--model_path', required=True, help='Path to pre-trained ArtDL model')
@click.option('--batch_size', default=32, help='Batch size for evaluation')
@click.option('--image_size', default=224, help='Input image size')
def main(model_path, batch_size, image_size):
    """ArtDL baseline evaluation on ICONCLASS and Wikidata datasets"""
    
    print(f"ArtDL Baseline Evaluation")
    print(f"Configuration:")
    print(f"  - Model path: {model_path}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Image size: {image_size}")
    print(f"  - Shared classes: {SHARED_CLASSES}")
    
    # Load pre-trained ArtDL model using simplified loading
    print(f"\nLoading pre-trained ArtDL model from {model_path}")
    model = load_pretrained_model(model_path, device, num_classes=10)
    model = model.to(device)
    model.eval()
    
    # Get transforms
    transform = get_artdl_transforms(image_size=image_size)
    
    # Evaluate on both datasets with both modalities
    datasets = ['ICONCLASS', 'wikidata']
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name} dataset")
        print(f"{'='*60}")
        
        # Full modality
        print(f"\n--- Full Modality (All Classes) ---")
        full_dataset = EvaluationDataset(dataset_name, shared_classes_only=False, 
                                       transform=transform, image_size=image_size)
        full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Create class mapping for full modality
        target_classes = [full_dataset.classes_df.iloc[i]['ID'] for i in range(len(full_dataset.classes_df))]
        full_class_mapping = create_class_mapping(None, target_classes)
        
        # Create output directory
        base_dir = os.path.dirname(__file__)
        full_output_dir = os.path.join(base_dir, 'full', dataset_name)
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Evaluate
        full_accuracy = evaluate_model(model, full_loader, dataset_name, full_output_dir, 
                                     full_class_mapping, "Full Classes")
        
        # Shared modality
        print(f"\n--- Shared Modality (Shared Classes Only) ---")
        shared_dataset = EvaluationDataset(dataset_name, shared_classes_only=True, 
                                         transform=transform, image_size=image_size)
        shared_loader = DataLoader(shared_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        # Create class mapping for shared modality
        shared_target_classes = [shared_dataset.classes_df.iloc[i]['ID'] for i in range(len(shared_dataset.classes_df))]
        shared_class_mapping = create_class_mapping(None, shared_target_classes)
        
        # Create output directory
        shared_output_dir = os.path.join(base_dir, 'shared', dataset_name)
        os.makedirs(shared_output_dir, exist_ok=True)
        
        # Evaluate
        shared_accuracy = evaluate_model(model, shared_loader, dataset_name, shared_output_dir, 
                                       shared_class_mapping, "Shared Classes")
        
        print(f"\n{dataset_name} Results:")
        print(f"  Full Classes Accuracy: {full_accuracy:.2f}%")
        print(f"  Shared Classes Accuracy: {shared_accuracy:.2f}%")
        print(f"  Full results saved to: {full_output_dir}")
        print(f"  Shared results saved to: {shared_output_dir}")
    
    print(f"\n{'='*60}")
    print(f"ArtDL Baseline Evaluation completed!")
    print(f"Results saved to baseline/full/ and baseline/shared/ directories")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
