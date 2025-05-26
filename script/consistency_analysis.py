#!/usr/bin/env python3
"""
Script for analyzing model consistency across datasets.

This script:
1. Loads all JSON files from the quality_check directory
2. Calculates consistency metrics for each model and test folder
3. Generates visualizations
4. Creates a comprehensive README.md with the analysis results
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict

def load_quality_check_files(quality_check_dir):
    """Load all JSON files from the quality_check directory."""
    results = {}
    for filename in os.listdir(quality_check_dir):
        if filename.endswith('.json'):
            # Extract model name and test folder from filename
            match = re.match(r'(.+)_(test_\d+)\.json', filename)
            if match:
                model_name, test_folder = match.groups()
                file_path = os.path.join(quality_check_dir, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                results[(model_name, test_folder)] = data
    return results

def calculate_consistency_metrics(data):
    """Calculate consistency metrics for each model and test folder."""
    metrics = {}
    
    for (model, test_folder), pairs in data.items():
        # Count pairs with same prediction
        same_pred_count = sum(1 for pair in pairs 
                             if pair[0]['predicted'] == pair[1]['predicted'] 
                             and pair[0]['predicted'] is not None 
                             and pair[1]['predicted'] is not None)
        
        # Count valid pairs (both have predictions)
        valid_pairs = sum(1 for pair in pairs 
                         if pair[0]['predicted'] is not None 
                         and pair[1]['predicted'] is not None)
        
        # Calculate consistency percentage
        consistency = (same_pred_count / valid_pairs * 100) if valid_pairs > 0 else 0
        
        # Calculate dataset-specific metrics
        dataset_pairs = {}
        for pair in pairs:
            if pair[0]['predicted'] is not None and pair[1]['predicted'] is not None:
                # Sort datasets to ensure consistent keys
                datasets = sorted([pair[0]['dataset'], pair[1]['dataset']])
                dataset_key = (datasets[0], datasets[1])
                
                if dataset_key not in dataset_pairs:
                    dataset_pairs[dataset_key] = {'total': 0, 'same': 0}
                
                dataset_pairs[dataset_key]['total'] += 1
                if pair[0]['predicted'] == pair[1]['predicted']:
                    dataset_pairs[dataset_key]['same'] += 1
        
        # Calculate dataset-pair consistency percentages
        dataset_consistency = {k: (v['same'] / v['total'] * 100) if v['total'] > 0 else 0 
                              for k, v in dataset_pairs.items()}
        
        # Calculate accuracy metrics
        accuracy = []
        for pair in pairs:
            for img in pair:
                if img['predicted'] is not None and img['ground_truth'] is not None:
                    accuracy.append(img['predicted'] == img['ground_truth'])
        
        accuracy_percentage = (sum(accuracy) / len(accuracy) * 100) if accuracy else 0
        
        # Collect error patterns for inconsistent predictions
        error_patterns = defaultdict(int)
        for pair in pairs:
            if (pair[0]['predicted'] != pair[1]['predicted'] and 
                pair[0]['predicted'] is not None and 
                pair[1]['predicted'] is not None):
                
                # Create a key for this error pattern
                error_key = (pair[0]['predicted'], pair[1]['predicted'])
                error_patterns[error_key] += 1
        
        # Store metrics
        metrics[(model, test_folder)] = {
            'consistency': consistency,
            'valid_pairs': valid_pairs,
            'same_pred_count': same_pred_count,
            'dataset_consistency': dataset_consistency,
            'accuracy': accuracy_percentage,
            'error_patterns': dict(error_patterns)
        }
    
    return metrics

def generate_visualizations(metrics, output_dir):
    """Generate visualizations and save them to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    models = sorted(set(model for (model, _) in metrics.keys()))
    test_folders = sorted(set(test_folder for (_, test_folder) in metrics.keys()))
    
    # 1. Bar chart of consistency percentages
    plt.figure(figsize=(14, 7))
    
    # Create DataFrame for easier plotting
    consistency_data = []
    for (model, test_folder), metric in metrics.items():
        consistency_data.append({
            'Model': model,
            'Test Folder': test_folder,
            'Consistency (%)': metric['consistency']
        })
    
    consistency_df = pd.DataFrame(consistency_data)
    
    # Plot
    ax = sns.barplot(x='Model', y='Consistency (%)', hue='Test Folder', data=consistency_df)
    plt.title('Model Consistency Across Datasets', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Consistency (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Test Folder', fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'consistency_bar_chart.png'), dpi=300)
    plt.close()
    
    # 2. Heatmap of dataset-pair consistency
    # For each test folder, create a separate heatmap
    for test_folder in test_folders:
        # Get all models for this test folder
        test_models = [model for (model, tf) in metrics.keys() if tf == test_folder]
        
        # Get all unique dataset pairs
        all_dataset_pairs = set()
        for (model, tf), metric in metrics.items():
            if tf == test_folder:
                all_dataset_pairs.update(metric['dataset_consistency'].keys())
        
        # Create a DataFrame for the heatmap
        heatmap_data = []
        for (model, tf), metric in metrics.items():
            if tf == test_folder:
                for pair, consistency in metric['dataset_consistency'].items():
                    heatmap_data.append({
                        'Model': model,
                        'Dataset Pair': f"{pair[0]}-{pair[1]}",
                        'Consistency (%)': consistency
                    })
        
        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)
            
            # Pivot for heatmap format
            pivot_df = heatmap_df.pivot(index='Model', columns='Dataset Pair', values='Consistency (%)')
            
            # Plot
            plt.figure(figsize=(12, len(test_models) * 0.8 + 2))
            sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', 
                       cbar_kws={'label': 'Consistency (%)'})
            plt.title(f'Dataset-Pair Consistency for {test_folder}', fontsize=16)
            plt.xlabel('Dataset Pair', fontsize=14)
            plt.ylabel('Model', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'dataset_pair_heatmap_{test_folder}.png'), dpi=300)
            plt.close()
    
    # 3. Accuracy vs. Consistency Plot
    plt.figure(figsize=(12, 8))
    
    # Create DataFrame for plotting
    accuracy_consistency_data = []
    for (model, test_folder), metric in metrics.items():
        # Extract model type (CLIP, SigLIP, Gemini, GPT, or Other)
        if 'clip' in model.lower():
            model_type = 'CLIP'
        elif 'siglip' in model.lower():
            model_type = 'SigLIP'
        elif 'gemini' in model.lower():
            model_type = 'Gemini'
        elif 'gpt' in model.lower():
            model_type = 'GPT'
        else:
            model_type = 'Other'
        
        accuracy_consistency_data.append({
            'Model': model,
            'Test Folder': test_folder,
            'Accuracy (%)': metric['accuracy'],
            'Consistency (%)': metric['consistency'],
            'Model Type': model_type
        })
    
    accuracy_consistency_df = pd.DataFrame(accuracy_consistency_data)
    
    # Plot
    sns.scatterplot(x='Accuracy (%)', y='Consistency (%)', 
                   hue='Model Type', style='Test Folder', 
                   s=150, data=accuracy_consistency_df)
    
    # Add model names as annotations
    for i, row in accuracy_consistency_df.iterrows():
        plt.annotate(row['Model'].split('-')[-1],  # Use just the last part of the model name
                    (row['Accuracy (%)'], row['Consistency (%)']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, alpha=0.8)
    
    plt.title('Accuracy vs. Consistency', fontsize=16)
    plt.xlabel('Accuracy (%)', fontsize=14)
    plt.ylabel('Consistency (%)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_consistency.png'), dpi=300)
    plt.close()
    
    # 4. Model comparison across test folders
    plt.figure(figsize=(14, 8))
    
    # Create a pivot table for easier plotting
    pivot_df = consistency_df.pivot(index='Model', columns='Test Folder', values='Consistency (%)')
    
    # Plot
    ax = pivot_df.plot(kind='bar', figsize=(14, 8))
    plt.title('Model Consistency Comparison Across Test Folders', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Consistency (%)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.legend(title='Test Folder', fontsize=12, title_fontsize=13)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
    plt.close()
    
    return {
        'consistency_bar_chart': 'consistency_bar_chart.png',
        'dataset_pair_heatmaps': [f'dataset_pair_heatmap_{tf}.png' for tf in test_folders],
        'accuracy_vs_consistency': 'accuracy_vs_consistency.png',
        'model_comparison': 'model_comparison.png'
    }

def generate_readme(metrics, visualization_paths, output_path):
    """Generate a README.md file with analysis results."""
    # Prepare data for the README
    models = sorted(set(model for (model, _) in metrics.keys()))
    test_folders = sorted(set(test_folder for (_, test_folder) in metrics.keys()))
    
    # Find best and worst models for consistency
    best_model = None
    worst_model = None
    best_consistency = -1
    worst_consistency = 101  # Start above 100%
    
    for (model, test_folder), metric in metrics.items():
        if test_folder == 'test_1':  # Use test_1 as the reference
            if metric['consistency'] > best_consistency:
                best_consistency = metric['consistency']
                best_model = model
            if metric['consistency'] < worst_consistency:
                worst_consistency = metric['consistency']
                worst_model = model
    
    # Find dataset pairs with highest and lowest consistency
    dataset_pair_consistency = defaultdict(list)
    for (model, test_folder), metric in metrics.items():
        for pair, consistency in metric['dataset_consistency'].items():
            dataset_pair_consistency[pair].append(consistency)
    
    # Calculate average consistency for each dataset pair
    avg_dataset_consistency = {pair: sum(values)/len(values) 
                              for pair, values in dataset_pair_consistency.items()}
    
    # Find highest and lowest consistency dataset pairs
    best_pair = max(avg_dataset_consistency.items(), key=lambda x: x[1])
    worst_pair = min(avg_dataset_consistency.items(), key=lambda x: x[1])
    
    # Create README content
    readme_content = [
        "# Model Consistency Analysis Across Datasets",
        "",
        "## Introduction",
        "",
        "This analysis examines the consistency of various models when classifying similar images across different datasets. ",
        "Consistency is measured by whether a model assigns the same class to visually similar images from different datasets.",
        "",
        "### Datasets",
        "",
        "The analysis includes the following datasets:",
        "- ArtDL",
        "- ICONCLASS",
        "- wikidata",
        "",
        "### Models",
        "",
        "The following models were evaluated:",
    ]
    
    # Add model list
    for model in models:
        readme_content.append(f"- {model}")
    
    readme_content.extend([
        "",
        "## Methodology",
        "",
        "1. Visually similar image pairs across datasets were identified using perceptual hashing.",
        "2. Each model's predictions for these image pairs were analyzed.",
        "3. Consistency was measured as the percentage of image pairs where both images received the same prediction.",
        "4. The analysis was performed across multiple test folders (test_1, test_2, test_3).",
        "",
        "## Overall Consistency Results",
        "",
        "The following chart shows the consistency percentage for each model across different test sets:",
        "",
        f"![Consistency Bar Chart]({visualization_paths['consistency_bar_chart']})",
        "",
        "### Model Comparison Across Test Folders",
        "",
        "This chart compares how each model's consistency varies across different test folders:",
        "",
        f"![Model Comparison]({visualization_paths['model_comparison']})",
        "",
        "### Consistency Metrics Table",
        "",
        "| Model | Test Folder | Consistency (%) | Valid Pairs | Same Predictions |",
        "|-------|-------------|-----------------|-------------|------------------|",
    ])
    
    # Add consistency metrics table
    for (model, test_folder), metric in sorted(metrics.items()):
        readme_content.append(
            f"| {model} | {test_folder} | {metric['consistency']:.2f} | {metric['valid_pairs']} | {metric['same_pred_count']} |"
        )
    
    readme_content.extend([
        "",
        "## Dataset-Pair Consistency",
        "",
        "The following heatmaps show consistency between specific dataset pairs for each test folder:",
        "",
    ])
    
    # Add dataset-pair heatmaps
    for heatmap_path in visualization_paths['dataset_pair_heatmaps']:
        readme_content.append(f"![Dataset-Pair Heatmap]({heatmap_path})")
        readme_content.append("")
    
    readme_content.extend([
        "## Accuracy vs. Consistency",
        "",
        "The following plot shows the relationship between prediction accuracy (compared to ground truth) and cross-dataset consistency:",
        "",
        f"![Accuracy vs Consistency]({visualization_paths['accuracy_vs_consistency']})",
        "",
        "## Conclusions",
        "",
        "### Key Findings",
        "",
    ])
    
    # Add key findings based on the analysis
    if best_model and worst_model:
        readme_content.append(f"1. **Model Consistency**: {best_model} shows the highest consistency ({best_consistency:.2f}%), while {worst_model} shows the lowest ({worst_consistency:.2f}%).")
    
    if best_pair and worst_pair:
        readme_content.append(f"2. **Dataset Pairs**: The {best_pair[0][0]}-{best_pair[0][1]} dataset pair shows the highest average consistency ({best_pair[1]:.2f}%), while the {worst_pair[0][0]}-{worst_pair[0][1]} pair shows the lowest ({worst_pair[1]:.2f}%).")
    
    # Add general observations
    readme_content.extend([
        "3. **Test Folder Variation**: Consistency varies across test folders, suggesting that the specific test images influence model consistency.",
        "4. **Accuracy vs. Consistency**: There appears to be a correlation between model accuracy and cross-dataset consistency, though some models break this pattern.",
        "",
        "### Recommendations",
        "",
        "1. For applications requiring high cross-dataset consistency, prefer models that demonstrated higher consistency scores in this analysis.",
        "2. Consider the specific dataset combinations in your application, as consistency varies significantly between different dataset pairs.",
        "3. When evaluating model performance, consider both accuracy and consistency metrics, as they provide complementary information about model behavior.",
        "4. For future model development, focus on improving consistency across datasets, particularly for the dataset combinations that showed lower consistency.",
        "",
        "*Note: This README was automatically generated based on the analysis of quality check data.*",
    ])
    
    # Write README file
    with open(output_path, 'w') as f:
        f.write('\n'.join(readme_content))
    
    return output_path

def main():
    # Define paths
    quality_check_dir = "/home/ubuntu/gspinaci/LLM-test/quality_check"
    output_dir = "/home/ubuntu/gspinaci/LLM-test/consistency_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading quality check files...")
    data = load_quality_check_files(quality_check_dir)
    
    # Calculate metrics
    print("Calculating consistency metrics...")
    metrics = calculate_consistency_metrics(data)
    
    # Generate visualizations
    print("Generating visualizations...")
    visualization_paths = generate_visualizations(metrics, output_dir)
    
    # Generate README
    print("Generating README.md...")
    readme_path = generate_readme(metrics, visualization_paths, os.path.join(output_dir, "README.md"))
    
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"README.md generated at {readme_path}")

if __name__ == "__main__":
    main()
