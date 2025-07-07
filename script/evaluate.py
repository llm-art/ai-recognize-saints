import os
import json
import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from PIL import Image
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, accuracy_score

# Suppress sklearn warnings about no positive class found
warnings.filterwarnings("ignore", message="No positive class found in y_true")

# Increase PIL's DecompressionBombWarning threshold to ~200 million pixels
Image.MAX_IMAGE_PIXELS = 200000000

KEY_CLASS_NAME = 'class_name'
KEY_NUM_IMAGES = '# of Images'
KEY_PRECISION = 'Precision'
KEY_RECALL = 'Recall'
KEY_F1_SCORE = 'F1 Score'
KEY_AVG_PRECISION = 'Average Precision'

def evaluate(model, images, classes, ground_truth_dict):
  # Load model data
  probs = np.load(os.path.join(model, 'probs.npy'))

  # Create confusion matrix using ground truth and predicted classes
  y_true = [ground_truth_dict.get(item) for item in images]
  y_pred = [classes[probs[i].argmax().item()][0] for i in range(min(len(images), len(probs)))]
  
  # Filter out classes that don't exist in the classes list
  class_ids = [t[0] for t in classes]
  
  # Create paired lists of valid true and predicted classes
  valid_pairs = []
  valid_indices = []  # Track indices of valid pairs
  for i, (true_cls, pred_cls) in enumerate(zip(y_true[:len(y_pred)], y_pred)):
      if true_cls in class_ids and pred_cls in class_ids:
          valid_pairs.append((true_cls, pred_cls))
          valid_indices.append(i)  # Store the index

  # Unzip the pairs into separate lists
  valid_y_true, valid_y_pred = zip(*valid_pairs) if valid_pairs else ([], [])
  
  y_true_indices = [next(i for i, t in enumerate(classes) if t[0] == cls) for cls in valid_y_true]
  y_pred_indices = [next(i for i, t in enumerate(classes) if t[0] == cls) for cls in valid_y_pred]

  # Filter probs to match the valid indices
  filtered_probs = probs[valid_indices] if valid_indices else np.array([])

  cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(classes)))

  acc = accuracy_score(y_true_indices, y_pred_indices) * 100

  confusion_matrix_df = pd.DataFrame(cm, index=[cls[0] for cls in classes], columns=[cls[0] for cls in classes])
  confusion_matrix_df.to_csv(os.path.join(model, 'confusion_matrix.csv'))

  plt.figure(figsize=(10, 8))
  sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Oranges', xticklabels=[cls[0] for cls in classes], yticklabels=[cls[0] for cls in classes])
  plt.xlabel('Predicted classes')
  plt.ylabel('Actual classes')
  plt.title(f'Confusion Matrix for {model}')
  plt.savefig(os.path.join(model, 'confusion_matrix.png'), bbox_inches='tight')
  plt.close()

  return y_true_indices, y_pred_indices, filtered_probs, acc

def class_metrics(y_true_indices, y_pred_indices, probs, classes, class_image_counts, model_path):
  y_true_one_hot = label_binarize(y_true_indices, classes=range(len(classes)))
  
  # Class-level metrics
  class_precisions = precision_score(y_true_indices, y_pred_indices, average=None, labels=range(len(classes)), zero_division=0) * 100
  class_recalls = recall_score(y_true_indices, y_pred_indices, average=None, labels=range(len(classes)), zero_division=0) * 100
  class_f1_scores = f1_score(y_true_indices, y_pred_indices, average=None, labels=range(len(classes)), zero_division=0) * 100
  class_avg_precisions = average_precision_score(y_true_one_hot, probs, average=None) * 100

  metrics_df = pd.DataFrame({
    KEY_CLASS_NAME: [cls[0] for cls in classes],
    KEY_NUM_IMAGES: [val for val in class_image_counts.values()],
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
  metrics_df.to_csv(os.path.join(model_path, 'class_metrics.csv'), index=False)

  # Save as Markdown
  metrics_df.to_markdown(os.path.join(model_path, 'class_metrics.md'), index=False)

  return mean_avg_precision, micro_avg_precision

@click.command()
@click.option('--models', multiple=True, default=['clip-vit-base-patch32', 'clip-vit-base-patch16', 'clip-vit-large-patch14', "siglip-base-patch16-512", "siglip-large-patch16-384", "siglip-so400m-patch14-384"], help='List of models to evaluate')
@click.option('--folders', multiple=True, default=['test_1', 'test_2', 'test_3'], help='List of folders to evaluate')
@click.option('--limit', default=-1, type=int, help='Limit the number of images to evaluate')
@click.option('--datasets', multiple=True, default=['ArtDL', 'IconArt'], help='Name of the dataset directory')
def main(models, folders, limit, datasets):
  base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
  
  for dataset in datasets:
    
    dataset_dir = os.path.join(base_dir, 'dataset', dataset)
    dataset_data_dir = os.path.join(base_dir, 'dataset', f'{dataset}-data')

    if not os.path.exists(dataset_dir):
      raise FileNotFoundError(f"Dataset not found at {dataset_dir}!")

    classes = {}
    ground_truth_dict = {item['item']: item['class'] for item in json.load(open(os.path.join(dataset_data_dir, '2_ground_truth.json'), 'r'))}
    images = open(os.path.join(dataset_data_dir, '2_test.txt'), 'r').read().splitlines()

    # Limit the number of images to evaluate
    if limit > 0:
      images = images[:limit]

    models = {folder: [os.path.join(base_dir, folder, dataset, model_name) for model_name in models] for folder in folders}
    
    classes_df = pd.read_csv(os.path.join(dataset_data_dir, 'classes.csv'))
    classes['test_1'] = list(classes_df[['ID', 'Label']].itertuples(index=False, name=None))
    classes['test_2'] = list(classes_df[['ID', 'Description']].itertuples(index=False, name=None))
    classes['test_3'] = list(classes_df[['ID', 'Label']].itertuples(index=False, name=None))
    classes['test_4'] = list(classes_df[['ID', 'Description']].itertuples(index=False, name=None))

    print(f"Dataset: {dataset}")

    for folder in folders:
      for model_path in models[folder]:
        
        print(f"{folder}, model: {model_path.split('/')[-1]}")

        # Perform evaluation
        y_true_indices, y_pred_indices, filtered_probs, acc = evaluate(model_path, images, classes[folder], ground_truth_dict)

        # Display accuracy in console
        print(f"Accuracy: {acc:.2f}%")

        # Class-level metrics
        class_image_counts = {cls[0]: y_true_indices.count(i) for i, cls in enumerate(classes[folder])}
        macro_avg_precision, micro_avg_precision = class_metrics(y_true_indices, y_pred_indices, filtered_probs, classes[folder], class_image_counts, model_path)

        # Store macro, micro average precision and accuracy
        summary_df = pd.DataFrame([{
            'Model': model_path.split('/')[-1],
            'Macro Average Precision': f"{macro_avg_precision:.2f}%",
            'Micro Average Precision': f"{micro_avg_precision:.2f}%",
            'Accuracy': f"{acc:.2f}%"
        }])

        summary_df.to_csv(os.path.join(model_path, 'summary_metrics.csv'), index=False)

if __name__ == '__main__':
  main()
