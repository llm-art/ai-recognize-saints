import os
import json
import click
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score, accuracy_score

KEY_EVALUATIONS = "evaluations"
KEY_CLASS_NAME = 'class_name'
KEY_NUM_IMAGES = '# of Images'
KEY_PRECISION = 'Precision'
KEY_RECALL = 'Recall'
KEY_F1_SCORE = 'F1 Score'
KEY_AVG_PRECISION = 'Average Precision'

def evaluate(model, images, classes, ground_truth_dict):
  # Load model data
  probs = torch.load(os.path.join(model, 'probs.pt'), weights_only=True)

  # Create confusion matrix using ground truth and predicted classes
  y_true = [ground_truth_dict.get(item) for item in images]
  y_pred = [classes[probs[i].argmax().item()][0] for i in range(len(images))]
  y_true_indices = [next(i for i, t in enumerate(classes) if t[0] == cls) for cls in y_true]
  y_pred_indices = [next(i for i, t in enumerate(classes) if t[0] == cls) for cls in y_pred]

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

  return y_true_indices, y_pred_indices, probs, acc

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

  # Reorder classes if needed
  class_order = [
    "11H(ANTONY OF PADUA)", "11H(FRANCIS)", "11H(JEROME)", 
    "11H(JOHN THE BAPTIST)", "11HH(MARY MAGDALENE)", "11H(PAUL)", 
    "11H(PETER)", "11H(DOMINIC)", "11H(SEBASTIAN)", "11F(MARY)"
  ]
  metrics_df[KEY_CLASS_NAME] = pd.Categorical(metrics_df[KEY_CLASS_NAME], categories=class_order, ordered=True)
  metrics_df = metrics_df.sort_values(KEY_CLASS_NAME).reset_index(drop=True)

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
@click.option('--models', multiple=True, help='List of models to evaluate')
@click.option('--folders', multiple=True, default=['test_1', 'test_2'], help='List of folders to evaluate')
@click.option('--limit', default=-1, type=int, help='Limit the number of images to evaluate')
def main(models, folders, limit):

  classes = {}
  ground_truth_dict = {item['item']: item['class'] for item in json.load(open(os.path.join('2_ground_truth.json'), 'r'))}
  images = open(os.path.join('2_test.txt'), 'r').read().splitlines()

  # Limit the number of images to evaluate
  if limit > 0:
    images = images[:limit]

  if not models:
    models = {folder: [os.path.join(folder, KEY_EVALUATIONS, name) for name in os.listdir(os.path.join(os.path.curdir, folder, KEY_EVALUATIONS)) if os.path.isdir(os.path.join(os.path.curdir, folder, KEY_EVALUATIONS, name))] for folder in folders}

  else:
    models = {folder: [os.path.join(folder, KEY_EVALUATIONS, model_name) for model_name in models] for folder in folders}

  
  classes_df = pd.read_csv('classes.csv')
  classes['test_1'] = list(classes_df[['ID', 'Label']].itertuples(index=False, name=None))
  classes['test_2'] = list(classes_df[['ID', 'Description']].itertuples(index=False, name=None))

  for folder in folders:
    for model_path in models[folder]:

      print(f"Model: {model_path}")

      # Perform evaluation
      y_true_indices, y_pred_indices, probs, acc = evaluate(model_path, images, classes[folder], ground_truth_dict)

      # Class-level metrics
      class_image_counts = {cls[0]: y_true_indices.count(i) for i, cls in enumerate(classes[folder])}
      macro_avg_precision, micro_avg_precision = class_metrics(y_true_indices, y_pred_indices, probs, classes[folder], class_image_counts, model_path)

      # Store macro, micro average precision and accuracy
      summary_df = pd.DataFrame([{
          'Model': model_path,
          'Macro Average Precision': f"{macro_avg_precision:.2f}%",
          'Micro Average Precision': f"{micro_avg_precision:.2f}%",
          'Accuracy': f"{acc:.2f}%"
      }])

      summary_df.to_csv(os.path.join(model_path, 'summary_metrics.csv'), index=False)

if __name__ == '__main__':
  main()