import os
import json
import click
import torch
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
KEY_ACCURACY = 'Accuracy'

def evaluate(model, images, classes, ground_truth_dict):
  # Load model data
  probs = torch.load(os.path.join(model, 'probs.pt'), weights_only=True)

  # Create confusion matrix using ground truth and predicted classes
  y_true = [ground_truth_dict.get(item) for item in images]
  y_pred = [classes[probs[i].argmax().item()][0] for i in range(len(images))]
  y_true_indices = [next(i for i, t in enumerate(classes) if t[0] == cls) for cls in y_true]
  y_pred_indices = [next(i for i, t in enumerate(classes) if t[0] == cls) for cls in y_pred]

  cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(classes)))

  confusion_matrix_df = pd.DataFrame(cm, index=[cls[0] for cls in classes], columns=[cls[0] for cls in classes])
  confusion_matrix_df.to_csv(os.path.join(model, 'confusion_matrix.csv'))

  plt.figure(figsize=(10, 8))
  sns.heatmap(confusion_matrix_df, annot=True, fmt='d', cmap='Oranges', xticklabels=[cls[0] for cls in classes], yticklabels=[cls[0] for cls in classes])
  plt.xlabel('Predicted classes')
  plt.ylabel('Actual classes')
  plt.title(f'Confusion Matrix for {model}')
  plt.savefig(os.path.join(model, 'confusion_matrix.png'), bbox_inches='tight')
  plt.close()

  return y_true_indices, y_pred_indices, probs

@click.command()
@click.option('--all', is_flag=True, help='Evaluate all')
def main(all):

  classes = {}
  ground_truth_dict = {item['item']: item['class'] for item in json.load(open(os.path.join('2_ground_truth.json'), 'r'))}
  images = open(os.path.join('2_test.txt'), 'r').read().splitlines()

  if all:
    folders = ['test_1', 'test_2']

  models = {folder: [os.path.join(folder, KEY_EVALUATIONS, name) for name in os.listdir(os.path.join(os.path.curdir, folder, KEY_EVALUATIONS)) if os.path.isdir(os.path.join(os.path.curdir, folder, KEY_EVALUATIONS, name))] for folder in folders}
  
  classes_df = pd.read_csv('classes.csv')
  classes['test_1'] = list(classes_df[['ID', 'Label']].itertuples(index=False, name=None))
  classes['test_2'] = list(classes_df[['ID', 'Description']].itertuples(index=False, name=None))

  for folder in folders:
    for model_path in models[folder]:

      print(f"Model: {model_path}")

      y_true_indices, y_pred_indices, probs = evaluate(model_path, images, classes[folder], ground_truth_dict)

      class_image_counts = {cls: y_true_indices.count(i) for i, cls in enumerate(classes[folder])}

      # Calculate metrics
      y_true_one_hot = label_binarize(y_true_indices, classes=range(len(classes[folder])))
      class_precisions = precision_score(y_true_indices, y_pred_indices, average=None, labels=range(len(classes[folder])), zero_division=0) * 100
      class_recalls = recall_score(y_true_indices, y_pred_indices, average=None, labels=range(len(classes[folder])), zero_division=0) * 100
      class_f1_scores = f1_score(y_true_indices, y_pred_indices, average=None, labels=range(len(classes[folder])), zero_division=0) * 100
      class_avg_precisions = average_precision_score(y_true_one_hot, probs, average=None) * 100

      # Create DF
      metrics_df = pd.DataFrame({
        KEY_CLASS_NAME: [cls[0] for cls in classes[folder]],
        KEY_NUM_IMAGES: [count for count in class_image_counts.values()],
        KEY_PRECISION: class_precisions,
        KEY_RECALL: class_recalls,
        KEY_F1_SCORE: class_f1_scores,
        KEY_AVG_PRECISION: class_avg_precisions
      })

      # Reorder classes
      class_order = ["11H(ANTONY OF PADUA)", "11H(FRANCIS)", "11H(JEROME)", "11H(JOHN THE BAPTIST)", "11HH(MARY MAGDALENE)", "11H(PAUL)", "11H(PETER)", "11H(DOMINIC)", "11H(SEBASTIAN)", "11F(MARY)"]
      metrics_df[KEY_CLASS_NAME] = pd.Categorical(metrics_df[KEY_CLASS_NAME], categories=class_order + ["MEAN"], ordered=True)
      metrics_df = metrics_df.sort_values(KEY_CLASS_NAME).reset_index(drop=True)

      # Add mean values to the dataframe
      mean_precision = precision_score(y_true_indices, y_pred_indices, average='macro', zero_division=0) * 100
      mean_recall = recall_score(y_true_indices, y_pred_indices, average='macro', zero_division=0) * 100
      mean_f1_score = f1_score(y_true_indices, y_pred_indices, average='macro', zero_division=0) * 100
      mean_avg_precision = average_precision_score(y_true_one_hot, probs, average='macro') * 100
      mean_values = {
          KEY_CLASS_NAME: 'Mean',
          KEY_NUM_IMAGES: '-',
          KEY_PRECISION: mean_precision,
          KEY_RECALL: mean_recall,
          KEY_F1_SCORE: mean_f1_score,
          KEY_AVG_PRECISION: mean_avg_precision
      }
      metrics_df = pd.concat([metrics_df, pd.DataFrame([mean_values])], ignore_index=True)
            
      metrics_df[[KEY_PRECISION, KEY_RECALL, KEY_F1_SCORE, KEY_AVG_PRECISION]] = metrics_df[
        [KEY_PRECISION, KEY_RECALL, KEY_F1_SCORE, KEY_AVG_PRECISION]
      ].map(lambda x: f"{x:.2f}%" if not isinstance(x, str) else x)
      metrics_df.to_csv(os.path.join(model_path, 'metrics.csv'), index=False)

if __name__ == '__main__':
  main()