#!/usr/bin/env python3
"""
Script to generate consistency data for different models and tests.

This script:
1. Loads image pairs from robust_cross_duplicates.json
2. For each model (siglip, clip, gpt) and test (1, 2, 3):
   - Creates a directory structure: data/consistency/{model}/{test}/
   - Loads ground truth data from the appropriate dataset's ground truth file
   - Gets model predictions for each image
   - Creates a JSON file with the same structure as cross_duplicates.json but with added ground_truth and predicted fields
   - Generates a readme file with tables for correctly and incorrectly predicted pairs
"""

import os
import json
import numpy as np
import shutil
from pathlib import Path

def load_pairs(file_path):
    """Load image pairs from the JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def load_ground_truth(dataset):
    """Load ground truth data for a specific dataset."""
    gt_path = f"/home/ubuntu/gspinaci/LLM-test/dataset/{dataset}-data/2_ground_truth.json"
    try:
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        return {item['item']: item['class'] for item in gt_data}
    except FileNotFoundError:
        print(f"Warning: Ground truth file not found for dataset {dataset}")
        return {}

def load_classes(dataset, test_folder):
    """Load class definitions for a dataset."""
    classes_path = f"/home/ubuntu/gspinaci/LLM-test/dataset/{dataset}-data/classes.csv"
    try:
        import pandas as pd
        classes_df = pd.read_csv(classes_path)
        
        if test_folder in ['test_1', 'test_3']:
            return list(classes_df[['ID', 'Label']].itertuples(index=False, name=None))
        else:  # test_2, test_4
            return list(classes_df[['ID', 'Description']].itertuples(index=False, name=None))
    except (FileNotFoundError, ImportError):
        print(f"Warning: Classes file not found for dataset {dataset} or pandas not available")
        return []

def get_prediction(image_name, model, test_folder, dataset):
    """Get the predicted class for an image using a specific model."""
    # Load test items to find the index of our image
    test_items_path = f"/home/ubuntu/gspinaci/LLM-test/dataset/{dataset}-data/2_test.txt"
    try:
        with open(test_items_path, 'r') as f:
            test_items = f.read().splitlines()
        
        if image_name not in test_items:
            return None
        
        image_index = test_items.index(image_name)
        
        # Load model predictions
        model_path = f"/home/ubuntu/gspinaci/LLM-test/{test_folder}/{dataset}/{model}"
        probs_path = os.path.join(model_path, 'probs.npy')
        
        if not os.path.exists(probs_path):
            return None
        
        # Load probabilities
        probs = np.load(probs_path)
        
        # Get classes for this dataset and test folder
        classes = load_classes(dataset, test_folder)
        
        # Get the predicted class (argmax of probabilities)
        if image_index < len(probs) and classes:
            pred_class_index = probs[image_index].argmax().item()
            return classes[pred_class_index][0]  # Return the class ID
    except Exception as e:
        print(f"Error getting prediction for {image_name}, {model}, {test_folder}, {dataset}: {e}")
    
    return None

def generate_readme(model, test_folder, processed_pairs, model_test_dir, output_dir):
    """Generate a readme file with tables for correctly and incorrectly predicted pairs."""
    # Identify correctly and incorrectly predicted pairs
    correct_pairs = []
    incorrect_pairs = []
    
    for pair in processed_pairs:
        # Skip pairs with missing predictions
        if pair[0]['predicted'] is None or pair[1]['predicted'] is None:
            continue
        
        # Check if predictions match
        if pair[0]['predicted'] == pair[1]['predicted']:
            correct_pairs.append(pair)
        else:
            incorrect_pairs.append(pair)
    
    # Create central example directory
    example_dir = os.path.join(output_dir, "example")
    os.makedirs(example_dir, exist_ok=True)
    
    # Process all pairs and prepare table data
    correct_table_data = []
    incorrect_table_data = []
    
    # Helper function to process a pair and add to the appropriate table data
    def process_pair(pair, is_correct):
        pair_data = []
        for j, img in enumerate(pair):
            # Extract image name and create destination path
            img_name = os.path.basename(img['path'])
            # Create a unique filename based on the image name
            dest_path = os.path.join(example_dir, f"image_{j+1}_{img_name}")
            
            # Copy image if it exists and it's not already copied
            if os.path.exists(img['path']) and not os.path.exists(dest_path):
                shutil.copy2(img['path'], dest_path)
            
            # Use path relative to the model_test_dir (where the readme file is located)
            if os.path.exists(dest_path):
                rel_path = os.path.relpath(dest_path, model_test_dir)
            else:
                rel_path = "Image not found"
            
            # Add to table data
            pair_data.append({
                "dataset": img['dataset'],
                "name": img['name'],
                "path": rel_path,
                "ground_truth": img['ground_truth'],
                "predicted": img['predicted']
            })
        
        if is_correct:
            correct_table_data.append(pair_data)
        else:
            incorrect_table_data.append(pair_data)
    
    # Process all pairs
    for pair in correct_pairs:
        process_pair(pair, True)
    
    for pair in incorrect_pairs:
        process_pair(pair, False)
    
    # Generate markdown tables
    markdown = f"## Image Pairs for {model} ({test_folder})\n\n"
    
    # Correct predictions table
    markdown += "### Correctly Predicted Pairs\n\n"
    markdown += "| Pair | Image 1 | Dataset | Ground Truth | Predicted | Image 2 | Dataset | Ground Truth | Predicted |\n"
    markdown += "|------|---------|---------|--------------|-----------|---------|---------|--------------|-----------|\n"
    
    for i, pair in enumerate(correct_table_data):
        img1 = pair[0]
        img2 = pair[1]
        markdown += f"| {i+1} | ![Image 1]({img1['path']}) | {img1['dataset']} | {img1['ground_truth']} | {img1['predicted']} | ![Image 2]({img2['path']}) | {img2['dataset']} | {img2['ground_truth']} | {img2['predicted']} |\n"
    
    # Incorrect predictions table
    markdown += "\n### Incorrectly Predicted Pairs\n\n"
    markdown += "| Pair | Image 1 | Dataset | Ground Truth | Predicted | Image 2 | Dataset | Ground Truth | Predicted |\n"
    markdown += "|------|---------|---------|--------------|-----------|---------|---------|--------------|-----------|\n"
    
    for i, pair in enumerate(incorrect_table_data):
        img1 = pair[0]
        img2 = pair[1]
        markdown += f"| {i+1} | ![Image 1]({img1['path']}) | {img1['dataset']} | {img1['ground_truth']} | {img1['predicted']} | ![Image 2]({img2['path']}) | {img2['dataset']} | {img2['ground_truth']} | {img2['predicted']} |\n"
    
    # Write markdown to file
    readme_path = os.path.join(model_test_dir, "readme.md")
    with open(readme_path, 'w') as f:
        f.write(markdown)
    
    print(f"Readme generated and saved to {readme_path}")
    print(f"Example images copied to {example_dir}")
    
    return readme_path

def process_model_test(model, test_folder, pairs, output_dir):
    """Process a specific model and test folder."""
    # Create output directory
    model_test_dir = os.path.join(output_dir, model, test_folder)
    os.makedirs(model_test_dir, exist_ok=True)
    
    # Load ground truth for all datasets to avoid loading multiple times
    ground_truths = {}
    datasets = set(img['dataset'] for pair in pairs for img in pair)
    for dataset in datasets:
        ground_truths[dataset] = load_ground_truth(dataset)
    
    # Process each pair
    processed_pairs = []
    for pair in pairs:
        processed_pair = []
        for img in pair:
            dataset = img['dataset']
            # Extract image name from path
            img_name = os.path.splitext(os.path.basename(img['path']))[0]
            
            # Get ground truth and prediction
            ground_truth = ground_truths.get(dataset, {}).get(img_name)
            predicted = get_prediction(img_name, model, test_folder, dataset)
            
            # Create processed image entry
            processed_img = img.copy()
            processed_img['ground_truth'] = ground_truth
            processed_img['predicted'] = predicted
            processed_pair.append(processed_img)
        
        processed_pairs.append(processed_pair)
    
    # Save results
    output_path = os.path.join(model_test_dir, 'consistency_data.json')
    with open(output_path, 'w') as f:
        json.dump(processed_pairs, f, indent=2)
    
    # Calculate and print statistics
    same_pred_count = sum(1 for pair in processed_pairs 
                         if pair[0]['predicted'] == pair[1]['predicted'] 
                         and pair[0]['predicted'] is not None 
                         and pair[1]['predicted'] is not None)
    valid_pairs = sum(1 for pair in processed_pairs 
                     if pair[0]['predicted'] is not None 
                     and pair[1]['predicted'] is not None)
    
    if valid_pairs > 0:
        print(f"{model} - {test_folder}: {same_pred_count}/{valid_pairs} pairs have same prediction ({same_pred_count/valid_pairs*100:.2f}%)")
    else:
        print(f"{model} - {test_folder}: No valid predictions found")
    
    # Generate readme file
    readme_path = generate_readme(model, test_folder, processed_pairs, model_test_dir, output_dir)
    
    return output_path, readme_path

def generate_summary(results, output_dir, clip_models, siglip_models, gpt_models, test_folders):
    """Generate a summary readme file with consistency statistics for all models and tests."""
    # Initialize results dictionary for summary
    summary_results = {}
    
    # Calculate consistency for each model and test
    for (model, test_folder), (data_path, _) in results.items():
        if model not in summary_results:
            summary_results[model] = {}
        
        # Extract consistency statistics from the results
        same_pred_count = 0
        valid_pairs = 0
        
        try:
            with open(data_path, 'r') as f:
                pairs = json.load(f)
            
            # Count pairs with same prediction
            same_pred_count = sum(1 for pair in pairs 
                                if pair[0]['predicted'] == pair[1]['predicted'] 
                                and pair[0]['predicted'] is not None 
                                and pair[1]['predicted'] is not None)
            
            # Count valid pairs (both images have predictions)
            valid_pairs = sum(1 for pair in pairs 
                            if pair[0]['predicted'] is not None 
                            and pair[1]['predicted'] is not None)
            
            if valid_pairs > 0:
                consistency = same_pred_count / valid_pairs * 100
            else:
                consistency = 0
        except Exception as e:
            print(f"Error calculating consistency for {data_path}: {e}")
            consistency = 0
            valid_pairs = 0
            same_pred_count = 0
        
        summary_results[model][test_folder] = (consistency, valid_pairs, same_pred_count)
    
    # Generate markdown content
    markdown = "# Consistency Analysis Summary\n\n"
    markdown += "This document provides a summary of the consistency statistics for all models and tests.\n\n"
    markdown += "Consistency is measured by the percentage of image pairs where both images receive the same prediction.\n\n"
    
    # CLIP Models table
    markdown += "## CLIP Models\n\n"
    markdown += "| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |\n"
    markdown += "|-------|------|-----------------|-------------|------------------|\n"
    
    for model in clip_models:
        for test_folder in test_folders:
            consistency, valid_pairs, same_pred_count = summary_results[model][test_folder]
            markdown += f"| {model} | {test_folder} | {consistency:.2f} | {valid_pairs} | {same_pred_count} |\n"
    
    # SigLIP Models table
    markdown += "\n## SigLIP Models\n\n"
    markdown += "| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |\n"
    markdown += "|-------|------|-----------------|-------------|------------------|\n"
    
    for model in siglip_models:
        for test_folder in test_folders:
            consistency, valid_pairs, same_pred_count = summary_results[model][test_folder]
            markdown += f"| {model} | {test_folder} | {consistency:.2f} | {valid_pairs} | {same_pred_count} |\n"
    
    # GPT Models table
    markdown += "\n## GPT Models\n\n"
    markdown += "| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |\n"
    markdown += "|-------|------|-----------------|-------------|------------------|\n"
    
    for model in gpt_models:
        for test_folder in test_folders:
            consistency, valid_pairs, same_pred_count = summary_results[model][test_folder]
            markdown += f"| {model} | {test_folder} | {consistency:.2f} | {valid_pairs} | {same_pred_count} |\n"
    
    # Key Observations
    markdown += "\n## Key Observations\n\n"
    
    # Find highest consistency for each test
    highest_test1 = max([(model, summary_results[model]['test_1'][0]) for model in summary_results], key=lambda x: x[1])
    highest_test2 = max([(model, summary_results[model]['test_2'][0]) for model in summary_results], key=lambda x: x[1])
    highest_test3 = max([(model, summary_results[model]['test_3'][0]) for model in summary_results], key=lambda x: x[1])
    
    # Find highest overall consistency
    highest_overall = max([(model, test, summary_results[model][test][0]) 
                          for model in summary_results 
                          for test in test_folders], 
                         key=lambda x: x[2])
    
    # Add observations
    markdown += f"- The {highest_overall[0]} model achieves the highest overall consistency ({highest_overall[2]:.2f}%) on {highest_overall[1]}.\n"
    markdown += f"- For test_1, the {highest_test1[0]} model shows the highest consistency ({highest_test1[1]:.2f}%).\n"
    markdown += f"- For test_2, the {highest_test2[0]} model shows the highest consistency ({highest_test2[1]:.2f}%).\n"
    markdown += f"- For test_3, the {highest_test3[0]} model shows the highest consistency ({highest_test3[1]:.2f}%).\n"
    
    # Add more observations based on the data
    gpt_variations = [(model, 
                      max([summary_results[model][test][0] for test in test_folders]) - 
                      min([summary_results[model][test][0] for test in test_folders])) 
                     for model in gpt_models]
    
    if gpt_variations:
        max_variation = max(gpt_variations, key=lambda x: x[1])
        markdown += f"- {max_variation[0]} shows extreme variation across tests: ranging from {min([summary_results[max_variation[0]][test][0] for test in test_folders]):.2f}% to {max([summary_results[max_variation[0]][test][0] for test in test_folders]):.2f}%.\n"
    
    # Write markdown to file
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(markdown)
    
    print(f"Summary readme generated and saved to {readme_path}")
    return readme_path

def main():
    # Define models and test folders
    clip_models = ['clip-vit-base-patch32', 'clip-vit-base-patch16', 'clip-vit-large-patch14']
    siglip_models = ['siglip-base-patch16-512', 'siglip-large-patch16-384', 'siglip-so400m-patch14-384']
    gpt_models = ['gpt-4o', 'gpt-4o-mini']
    
    all_models = clip_models + siglip_models + gpt_models
    test_folders = ['test_1', 'test_2', 'test_3']
    
    # Load image pairs
    pairs_path = "/home/ubuntu/gspinaci/LLM-test/dataset/analysis/robust_cross_duplicates.json"
    pairs = load_pairs(pairs_path)
    
    # Create output directory
    output_dir = "/home/ubuntu/gspinaci/LLM-test/dataset/consistency"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each model and test folder
    results = {}
    for model in all_models:
        for test_folder in test_folders:
            output_path, readme_path = process_model_test(model, test_folder, pairs, output_dir)
            results[(model, test_folder)] = (output_path, readme_path)
    
    # Generate summary readme
    summary_path = generate_summary(results, output_dir, clip_models, siglip_models, gpt_models, test_folders)
    
    print(f"\nProcessing complete! Results saved to {output_dir}")
    print("Generated files:")
    for (model, test_folder), (data_path, readme_path) in results.items():
        print(f"  - {model}/{test_folder}:")
        print(f"    - Data: {data_path}")
        print(f"    - Readme: {readme_path}")
    print(f"  - Summary: {summary_path}")

if __name__ == "__main__":
    main()
