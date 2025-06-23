import os
import json
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class IconographyDataset(Dataset):
    """Dataset class for ICONCLASS and wikidata datasets with 80/20 train/test split"""
    def __init__(self, dataset_name, mode='train', transform=None, train_split=0.8, image_size=224):
        self.dataset_name = dataset_name
        self.mode = mode
        self.transform = transform
        self.image_size = image_size
        
        # Set paths
        base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        self.data_dir = os.path.join(base_dir, 'dataset', f'{dataset_name}-data')
        self.images_dir = os.path.join(base_dir, 'dataset', dataset_name, 'JPEGImages')
        
        # Load ground truth and classes
        with open(os.path.join(self.data_dir, '2_ground_truth.json'), 'r') as f:
            ground_truth_list = json.load(f)
            # Convert list format to dictionary
            self.ground_truth = {item['item']: item['class'] for item in ground_truth_list}
        
        self.classes_df = pd.read_csv(os.path.join(self.data_dir, 'classes.csv'))
        self.class_to_idx = {row['ID']: idx for idx, row in self.classes_df.iterrows()}
        self.idx_to_class = {idx: row['ID'] for idx, row in self.classes_df.iterrows()}
        
        # Get all available images from ground truth
        all_images = list(self.ground_truth.keys())
        
        # Filter images that have valid classes
        valid_classes = set(self.class_to_idx.keys())
        all_images = [img for img in all_images 
                     if img in self.ground_truth and self.ground_truth[img] in valid_classes]
        
        # Create 80/20 train/test split
        if mode == 'train':
            self.image_list, _ = train_test_split(
                all_images, 
                train_size=train_split, 
                random_state=42, 
                stratify=[self.ground_truth[img] for img in all_images]
            )
        else:  # test mode
            _, self.image_list = train_test_split(
                all_images, 
                train_size=train_split, 
                random_state=42, 
                stratify=[self.ground_truth[img] for img in all_images]
            )
        
        # Filter out images that cannot be loaded
        self._filter_valid_images()
        
        print(f"Loaded {len(self.image_list)} images for {dataset_name} {mode} set")
        print(f"Using {len(valid_classes)} classes: {list(valid_classes)}")
    
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
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        
        # Try different image extensions
        possible_extensions = ['.jpg', '.jpeg', '.png', '']
        image = None
        img_path = None
        
        for ext in possible_extensions:
            img_path = os.path.join(self.images_dir, img_name + ext)
            if os.path.exists(img_path):
                try:
                    image = Image.open(img_path).convert('RGB')
                    break
                except Exception as e:
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
            # Default transform if none provided
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        
        return image, label, img_name

def get_transforms(mode='train', image_size=224):
    """Get standard transforms for vanilla ResNet50"""
    
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_weighted_sampler(dataset):
    """Create weighted sampler for handling class imbalance"""
    # Count samples per class
    class_counts = {}
    for _, label, _ in dataset:
        class_id = dataset.idx_to_class[label]
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    # Calculate weights (inverse frequency weighting)
    total_samples = len(dataset)
    class_weights = {class_id: total_samples / (len(class_counts) * count) 
                    for class_id, count in class_counts.items()}
    
    # Create sample weights
    sample_weights = []
    for _, label, _ in dataset:
        class_id = dataset.idx_to_class[label]
        sample_weights.append(class_weights[class_id])
    
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def create_vanilla_resnet50(num_classes, pretrained=True):
    """Create vanilla ResNet50 model"""
    model = models.resnet50(pretrained=pretrained)
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, train_loader, num_epochs, learning_rate, save_path):
    """Train vanilla ResNet50 model on training set only"""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training history
    train_losses = []
    train_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels, _ in train_pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        
        # Save model every 10 epochs and at the end
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), save_path)
            print(f'Model saved at epoch {epoch+1}')
        
        # Learning rate scheduling
        scheduler.step()
        
        print('-' * 50)
    
    # Plot training curves
    plot_training_curves(train_losses, train_accs, os.path.dirname(save_path))
    
    return model

def plot_training_curves(train_losses, train_accs, save_dir):
    """Plot and save training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, test_loader, dataset_name, save_dir):
    """Evaluate model and save results"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_names = []
    
    with torch.no_grad():
        for images, labels, names in tqdm(test_loader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_names.extend(names)
    
    # Save predictions and probabilities
    np.save(os.path.join(save_dir, 'probs.npy'), np.array(all_probs))
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    # Create confusion matrix
    dataset = test_loader.dataset
    class_names = [dataset.classes_df.iloc[i]['ID'] for i in range(len(dataset.classes_df))]
    class_labels = [dataset.classes_df.iloc[i]['Label'] for i in range(len(dataset.classes_df))]
    
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    
    # Save confusion matrix as CSV
    confusion_matrix_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    confusion_matrix_df.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted classes')
    plt.ylabel('Actual classes')
    plt.title(f'Confusion Matrix for {dataset_name} (Vanilla ResNet50)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate class metrics
    class_metrics(all_labels, all_preds, np.array(all_probs), dataset, save_dir, accuracy)
    
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
    
    # Class-level metrics
    class_precisions = precision_score(y_true_indices, y_pred_indices, average=None, 
                                     labels=range(len(classes)), zero_division=0) * 100
    class_recalls = recall_score(y_true_indices, y_pred_indices, average=None, 
                               labels=range(len(classes)), zero_division=0) * 100
    class_f1_scores = f1_score(y_true_indices, y_pred_indices, average=None, 
                             labels=range(len(classes)), zero_division=0) * 100
    class_avg_precisions = average_precision_score(y_true_one_hot, probs, average=None) * 100
    
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
    mean_avg_precision = average_precision_score(y_true_one_hot, probs, average='macro') * 100
    
    micro_precision = precision_score(y_true_indices, y_pred_indices, average='micro', zero_division=0) * 100
    micro_recall = recall_score(y_true_indices, y_pred_indices, average='micro', zero_division=0) * 100
    micro_f1_score = f1_score(y_true_indices, y_pred_indices, average='micro', zero_division=0) * 100
    micro_avg_precision = average_precision_score(y_true_one_hot, probs, average='micro') * 100
    
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
        'Model': 'vanilla-resnet50',
        'Macro Average Precision': f"{mean_avg_precision:.2f}%",
        'Micro Average Precision': f"{micro_avg_precision:.2f}%",
        'Accuracy': f"{accuracy:.2f}%"
    }])
    
    summary_df.to_csv(os.path.join(save_dir, 'summary_metrics.csv'), index=False)
    
    return mean_avg_precision, micro_avg_precision

@click.command()
@click.option('--dataset', type=click.Choice(['ICONCLASS', 'wikidata']), required=True,
              help='Dataset to train on')
@click.option('--epochs', default=50, help='Number of training epochs')
@click.option('--batch_size', default=32, help='Batch size')
@click.option('--learning_rate', default=1e-3, help='Learning rate')
@click.option('--image_size', default=224, help='Input image size')
@click.option('--pretrained', default=True, help='Use ImageNet pretrained weights')
@click.option('--weighted_sampling', default=True, help='Use weighted sampling for class imbalance')
def main(dataset, epochs, batch_size, learning_rate, image_size, pretrained, weighted_sampling):
    """Vanilla ResNet50 trainer for ICONCLASS and Wikidata datasets"""
    
    print(f"Training vanilla ResNet50 on {dataset} dataset")
    print(f"Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Image size: {image_size}")
    print(f"  - Pretrained: {pretrained}")
    print(f"  - Weighted sampling: {weighted_sampling}")
    
    # Create output directory
    base_dir = os.path.dirname(__file__)
    output_dir = os.path.join(base_dir, 'training', dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of classes (both datasets have 10 classes)
    num_classes = 10
    
    # Create datasets with 80/20 split
    train_transform = get_transforms(mode='train', image_size=image_size)
    test_transform = get_transforms(mode='test', image_size=image_size)
    
    train_dataset = IconographyDataset(dataset, mode='train', transform=train_transform, train_split=0.8)
    test_dataset = IconographyDataset(dataset, mode='test', transform=test_transform, train_split=0.8)
    
    # Create data loaders
    if weighted_sampling:
        train_sampler = create_weighted_sampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create vanilla ResNet50 model
    model = create_vanilla_resnet50(num_classes, pretrained=pretrained)
    
    # Train model
    model_save_path = os.path.join(output_dir, 'trained_model.pth')
    trained_model = train_model(model, train_loader, epochs, learning_rate, model_save_path)
    
    # Load trained model for evaluation
    model.load_state_dict(torch.load(model_save_path))
    model = model.to(device)
    
    # Evaluate on test set
    print(f"\nEvaluating on {dataset} test set...")
    accuracy = evaluate_model(model, test_loader, dataset, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Dataset: {dataset}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
