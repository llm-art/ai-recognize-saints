#!/usr/bin/env python3
"""
Script for analyzing image dataset overlap, duplicate detection, and benchmark preparation.

This script provides functionality for:
1. Extracting CLIP embeddings from test datasets
2. Detecting duplicates using perceptual hashing or cosine similarity
3. Projecting embeddings using t-SNE and UMAP
4. Preparing few-shot benchmarks with various sampling strategies
5. Evaluating cross-dataset generalization
"""

import os
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
import imagehash
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import pairwise_distances
import umap
import json
import logging
import random
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib_venn import venn2, venn3
import click
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===============================
# Data Loading Utilities
# ===============================

def load_dataset_info(dataset_name, base_dir, use_ground_truth=False):
    """
    Load dataset information including image paths and labels.
    
    Args:
        dataset_name (str): Name of the dataset
        base_dir (str): Base directory of the project
        use_ground_truth (bool): Whether to use ground truth labels from 2_ground_truth.json
        
    Returns:
        dict: Dictionary containing dataset information
    """
    dataset_data_dir = os.path.join(base_dir, 'dataset', f'{dataset_name}-data')
    
    # Load test image list
    with open(os.path.join(dataset_data_dir, '2_test.txt'), 'r') as file:
        image_list = file.read().splitlines()
    
    # Try to load class labels from classes.csv
    classes_path = os.path.join(dataset_data_dir, 'classes.csv')
    class_mapping = {}
    if os.path.exists(classes_path):
        try:
            classes_df = pd.read_csv(classes_path)
            # Create mapping from ID to Label
            class_mapping = dict(zip(classes_df['ID'], classes_df['Label']))
            logger.info(f"Loaded {len(class_mapping)} class labels from {classes_path}")
        except Exception as e:
            logger.error(f"Error loading classes from {classes_path}: {e}")
    
    # Load ground truth labels if requested
    if use_ground_truth:
        ground_truth_path = os.path.join(dataset_data_dir, '2_ground_truth.json')
        if os.path.exists(ground_truth_path):
            try:
                with open(ground_truth_path, 'r') as file:
                    ground_truth = json.load(file)
                # Create mapping from item to class
                ground_truth_labels = {item['item']: item['class'] for item in ground_truth}
                # Apply to our image list
                labels = {img: ground_truth_labels.get(img, "unknown") for img in image_list}
                logger.info(f"Loaded {len(ground_truth_labels)} ground truth labels from {ground_truth_path}")
            except Exception as e:
                logger.error(f"Error loading ground truth from {ground_truth_path}: {e}")
                # Fall back to regular label loading
                labels = load_regular_labels(image_list, dataset_data_dir, class_mapping)
        else:
            logger.warning(f"Ground truth file {ground_truth_path} not found, using fallback labels")
            # Fall back to regular label loading
            labels = load_regular_labels(image_list, dataset_data_dir, class_mapping)
    else:
        # Use regular label loading
        labels = load_regular_labels(image_list, dataset_data_dir, class_mapping)
    
    # Get image paths
    image_dir = os.path.join(base_dir, 'dataset', dataset_name, 'JPEGImages')
    image_paths = [os.path.join(image_dir, f"{img}.jpg") for img in image_list]
    
    # Create analysis directory
    analysis_dir = os.path.join(dataset_data_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    return {
        "name": dataset_name,
        "image_list": image_list,
        "image_paths": image_paths,
        "labels": labels,
        "analysis_dir": analysis_dir
    }

def load_regular_labels(image_list, dataset_data_dir, class_mapping):
    """
    Load labels using the regular method (from labels.json or class mapping).
    
    Args:
        image_list (list): List of image names
        dataset_data_dir (str): Path to the dataset data directory
        class_mapping (dict): Mapping from class IDs to labels
        
    Returns:
        dict: Dictionary mapping image names to labels
    """
    # Load labels if available
    labels_path = os.path.join(dataset_data_dir, 'labels.json')
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as file:
            labels = json.load(file)
        logger.info(f"Loaded labels from {labels_path}")
    else:
        # Create labels based on class mapping or use "unknown"
        labels = {}
        for img in image_list:
            # Extract class ID from image name (assuming format like "class_id/image_name")
            parts = img.split('/')
            if len(parts) > 1:
                class_id = parts[0]
                # Use class label from mapping if available
                labels[img] = class_mapping.get(class_id, "unknown")
            else:
                labels[img] = "unknown"
        logger.info(f"Created labels from class mapping for {len(image_list)} images")
    
    return labels

def load_images(dataset_info, max_images=None):
    """
    Load images from disk given dataset information.
    
    Args:
        dataset_info (dict): Dataset information from load_dataset_info
        max_images (int, optional): Maximum number of images to load
        
    Returns:
        list: List of tuples (image_name, PIL.Image, label)
    """
    image_list = dataset_info["image_list"]
    image_paths = dataset_info["image_paths"]
    labels = dataset_info["labels"]
    
    # Limit number of images if specified
    if max_images is not None:
        image_list = image_list[:max_images]
        image_paths = image_paths[:max_images]
    
    images = []
    for img_name, img_path in tqdm(zip(image_list, image_paths), 
                                  desc=f"Loading {dataset_info['name']} images",
                                  total=len(image_list)):
        try:
            image = Image.open(img_path).convert("RGB")
            label = labels.get(img_name, "unknown")
            images.append((img_name, image, label))
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
    
    return images

# ===============================
# Feature Extraction
# ===============================

def extract_clip_embeddings(images, model, preprocess, device, batch_size=32):
    """
    Extract CLIP embeddings from images.
    
    Args:
        images (list): List of tuples (image_name, PIL.Image, label)
        model: CLIP model
        preprocess: CLIP preprocessing function
        device: PyTorch device
        batch_size (int): Batch size for processing
        
    Returns:
        tuple: (image_names, embeddings, labels)
    """
    all_embeddings = []
    all_names = []
    all_labels = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Extracting CLIP embeddings"):
        batch = images[i:i+batch_size]
        names = [item[0] for item in batch]
        imgs = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        
        # Preprocess images
        image_input = torch.stack([preprocess(img) for img in imgs]).to(device)
        
        # Extract features
        with torch.no_grad():
            image_embeddings = model.encode_image(image_input)
            # Normalize embeddings
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        
        all_embeddings.append(image_embeddings.cpu().numpy())
        all_names.extend(names)
        all_labels.extend(labels)
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    return all_names, all_embeddings, all_labels

def compute_perceptual_hashes(images, hash_size=8, hash_type='phash'):
    """
    Compute perceptual hashes for images.
    
    Args:
        images (list): List of tuples (image_name, PIL.Image, label)
        hash_size (int): Size of the hash
        hash_type (str): Type of hash ('phash', 'dhash', 'whash', 'ahash')
        
    Returns:
        tuple: (image_names, hashes, labels)
    """
    all_names = []
    all_hashes = []
    all_labels = []
    
    hash_functions = {
        'phash': imagehash.phash,
        'dhash': imagehash.dhash,
        'whash': imagehash.whash,
        'ahash': imagehash.average_hash
    }
    
    hash_func = hash_functions.get(hash_type, imagehash.phash)
    
    for name, image, label in tqdm(images, desc=f"Computing {hash_type}"):
        try:
            img_hash = hash_func(image, hash_size=hash_size)
            all_names.append(name)
            all_hashes.append(img_hash)
            all_labels.append(label)
        except Exception as e:
            logger.error(f"Error computing hash for image {name}: {e}")
    
    return all_names, all_hashes, all_labels

# ===============================
# Duplicate Detection
# ===============================

def detect_duplicates_cosine(embeddings, names, threshold=0.9):
    """
    Detect duplicate images using cosine similarity of embeddings.
    
    Args:
        embeddings (numpy.ndarray): Image embeddings
        names (list): Image names
        threshold (float): Similarity threshold for considering duplicates
        
    Returns:
        list: List of duplicate pairs (name1, name2, similarity)
    """
    # Compute pairwise cosine similarities
    similarity_matrix = np.matmul(embeddings, embeddings.T)
    
    # Find duplicates (pairs with similarity > threshold)
    duplicates = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                duplicates.append((names[i], names[j], float(similarity)))
    
    return duplicates

def detect_duplicates_perceptual(hashes, names, threshold=5):
    """
    Detect duplicate images using perceptual hash similarity.
    
    Args:
        hashes (list): List of perceptual hashes
        names (list): Image names
        threshold (int): Maximum hash difference for considering duplicates
        
    Returns:
        list: List of duplicate pairs (name1, name2, difference)
    """
    duplicates = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            difference = hashes[i] - hashes[j]
            if difference <= threshold:
                duplicates.append((names[i], names[j], int(difference)))
    
    return duplicates

def detect_cross_dataset_duplicates(embeddings1, names1, embeddings2, names2, threshold=0.9):
    """
    Detect duplicate images across two datasets using cosine similarity.
    
    Args:
        embeddings1 (numpy.ndarray): Embeddings from first dataset
        names1 (list): Image names from first dataset
        embeddings2 (numpy.ndarray): Embeddings from second dataset
        names2 (list): Image names from second dataset
        threshold (float): Similarity threshold for considering duplicates
        
    Returns:
        list: List of duplicate pairs (name1, name2, similarity)
    """
    # Compute cross-dataset similarity matrix
    similarity_matrix = np.matmul(embeddings1, embeddings2.T)
    
    # Find duplicates (pairs with similarity > threshold)
    duplicates = []
    for i in range(len(names1)):
        for j in range(len(names2)):
            similarity = similarity_matrix[i, j]
            if similarity > threshold:
                duplicates.append((names1[i], names2[j], float(similarity)))
    
    return duplicates

def detect_cross_dataset_duplicates_perceptual(hashes1, names1, hashes2, names2, threshold=5):
    """
    Detect duplicate images across two datasets using perceptual hash similarity.
    
    Args:
        hashes1 (list): Perceptual hashes from first dataset
        names1 (list): Image names from first dataset
        hashes2 (list): Perceptual hashes from second dataset
        names2 (list): Image names from second dataset
        threshold (int): Maximum hash difference for considering duplicates
        
    Returns:
        list: List of duplicate pairs (name1, name2, difference)
    """
    duplicates = []
    for i in range(len(names1)):
        for j in range(len(names2)):
            difference = hashes1[i] - hashes2[j]
            if difference <= threshold:
                duplicates.append((names1[i], names2[j], int(difference)))
    
    return duplicates

# ===============================
# Embedding Projection
# ===============================

def project_embeddings_tsne(embeddings, names, labels, perplexity=30, n_iter=1000, random_state=42):
    """
    Project embeddings to 2D using t-SNE.
    
    Args:
        embeddings (numpy.ndarray): Image embeddings
        names (list): Image names
        labels (list): Image labels
        perplexity (int): t-SNE perplexity parameter
        n_iter (int): Number of iterations
        random_state (int): Random seed
        
    Returns:
        pandas.DataFrame: DataFrame with projected coordinates and metadata
    """
    logger.info(f"Computing t-SNE projection with perplexity={perplexity}...")
    
    # Adjust perplexity if needed
    perplexity = min(perplexity, len(embeddings) - 1)
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state
    )
    tsne_result = tsne.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': tsne_result[:, 0],
        'y': tsne_result[:, 1],
        'name': names,
        'label': labels
    })
    
    return df

def project_embeddings_umap(embeddings, names, labels, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Project embeddings to 2D using UMAP.
    
    Args:
        embeddings (numpy.ndarray): Image embeddings
        names (list): Image names
        labels (list): Image labels
        n_neighbors (int): UMAP n_neighbors parameter
        min_dist (float): UMAP min_dist parameter
        random_state (int): Random seed
        
    Returns:
        pandas.DataFrame: DataFrame with projected coordinates and metadata
    """
    logger.info(f"Computing UMAP projection with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    
    # Adjust n_neighbors if needed
    n_neighbors = min(n_neighbors, len(embeddings) - 1)
    
    # Apply UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    umap_result = reducer.fit_transform(embeddings)
    
    # Create DataFrame
    df = pd.DataFrame({
        'x': umap_result[:, 0],
        'y': umap_result[:, 1],
        'name': names,
        'label': labels
    })
    
    return df

def plot_projection(df, title, output_path, color_by='label', hover_data=None, projection_type='t-SNE'):
    """
    Plot projection results.
    
    Args:
        df (pandas.DataFrame): DataFrame with projection results
        title (str): Plot title
        output_path (str): Path to save the plot
        color_by (str): Column to use for coloring points
        hover_data (list): Additional columns to show in hover data
        projection_type (str): Type of projection ('t-SNE' or 'UMAP')
    """
    if hover_data is None:
        hover_data = ['name']
    
    # Create interactive plot with Plotly
    fig = px.scatter(
        df, x='x', y='y', color=color_by, hover_data=hover_data,
        title=title,
        labels={
            'x': f'{projection_type} Dimension 1', 
            'y': f'{projection_type} Dimension 2', 
            color_by: color_by.capitalize()
        }
    )
    
    fig.update_layout(
        width=1000, 
        height=800,
        xaxis_title=f"{projection_type} Dimension 1 (CLIP embedding projection)",
        yaxis_title=f"{projection_type} Dimension 2 (CLIP embedding projection)"
    )
    
    # Save as HTML for interactivity
    pio.write_html(fig, output_path)
    
    # Also save as PNG for quick viewing
    png_path = output_path.replace('.html', '.png')
    fig.write_image(png_path)
    
    logger.info(f"Projection plot saved to {output_path} and {png_path}")

def plot_combined_projection(datasets, projection_type, output_dir):
    """
    Create a combined projection plot with data from multiple datasets.
    
    Args:
        datasets (list): List of dictionaries with dataset information
        projection_type (str): Type of projection ('tsne' or 'umap')
        output_dir (str): Directory to save the plot
    """
    # Combine embeddings and metadata
    all_embeddings = []
    all_names = []
    all_labels = []
    all_sources = []
    
    for dataset in datasets:
        all_embeddings.append(dataset['embeddings'])
        all_names.extend(dataset['names'])
        all_labels.extend(dataset['labels'])
        all_sources.extend([dataset['name']] * len(dataset['names']))
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    
    # Project embeddings
    if projection_type == 'tsne':
        df = project_embeddings_tsne(all_embeddings, all_names, all_labels)
    else:  # umap
        df = project_embeddings_umap(all_embeddings, all_names, all_labels)
    
    # Add source information
    df['source'] = all_sources
    
    # Check if any dataset is using ground truth labels
    using_ground_truth = any(hasattr(dataset.get('info', {}), 'get') and 
                            dataset.get('info', {}).get('using_ground_truth', False) 
                            for dataset in datasets)
    
    # Add label source to title
    label_source = "Ground Truth Labels" if using_ground_truth else "Default Labels"
    
    # Plot by dataset source
    plot_projection(
        df, 
        f'Combined {projection_type.upper()} Projection by Dataset ({label_source})',
        os.path.join(output_dir, f'combined_{projection_type}_by_dataset.html'),
        color_by='source',
        hover_data=['name', 'label']
    )
    
    # Plot by label
    plot_projection(
        df, 
        f'Combined {projection_type.upper()} Projection by Label ({label_source})',
        os.path.join(output_dir, f'combined_{projection_type}_by_label.html'),
        color_by='label',
        hover_data=['name', 'source']
    )

# ===============================
# Few-shot Sampling Strategies
# ===============================

def random_sampling(names, labels, k_shot, n_classes, seed=42):
    """
    Perform random k-shot sampling.
    
    Args:
        names (list): Image names
        labels (list): Image labels
        k_shot (int): Number of examples per class
        n_classes (int): Number of classes to sample
        seed (int): Random seed
        
    Returns:
        tuple: (sampled_names, sampled_labels)
    """
    random.seed(seed)
    
    # Get unique labels
    unique_labels = list(set(labels))
    
    # If n_classes is greater than available classes, use all available
    n_classes = min(n_classes, len(unique_labels))
    
    # Randomly select n_classes
    selected_classes = random.sample(unique_labels, n_classes)
    
    # Sample k examples from each selected class
    sampled_names = []
    sampled_labels = []
    
    for cls in selected_classes:
        # Get all examples of this class
        cls_indices = [i for i, label in enumerate(labels) if label == cls]
        
        # If we have fewer than k examples, use all available
        k = min(k_shot, len(cls_indices))
        
        # Randomly sample k examples
        sampled_indices = random.sample(cls_indices, k)
        
        # Add to our samples
        sampled_names.extend([names[i] for i in sampled_indices])
        sampled_labels.extend([labels[i] for i in sampled_indices])
    
    return sampled_names, sampled_labels

def stratified_sampling(names, labels, k_shot, n_classes, seed=42):
    """
    Perform stratified k-shot sampling (balanced class distribution).
    
    Args:
        names (list): Image names
        labels (list): Image labels
        k_shot (int): Number of examples per class
        n_classes (int): Number of classes to sample
        seed (int): Random seed
        
    Returns:
        tuple: (sampled_names, sampled_labels)
    """
    random.seed(seed)
    
    # Count examples per class
    class_counts = Counter(labels)
    
    # Sort classes by frequency (descending)
    sorted_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)
    
    # If n_classes is greater than available classes, use all available
    n_classes = min(n_classes, len(sorted_classes))
    
    # Select the n_classes most frequent classes
    selected_classes = sorted_classes[:n_classes]
    
    # Sample k examples from each selected class
    sampled_names = []
    sampled_labels = []
    
    for cls in selected_classes:
        # Get all examples of this class
        cls_indices = [i for i, label in enumerate(labels) if label == cls]
        
        # If we have fewer than k examples, use all available
        k = min(k_shot, len(cls_indices))
        
        # Randomly sample k examples
        sampled_indices = random.sample(cls_indices, k)
        
        # Add to our samples
        sampled_names.extend([names[i] for i in sampled_indices])
        sampled_labels.extend([labels[i] for i in sampled_indices])
    
    return sampled_names, sampled_labels

def long_tail_sampling(names, labels, k_shot, n_classes, seed=42, alpha=0.5):
    """
    Perform long-tail aware k-shot sampling (focus on underrepresented classes).
    
    Args:
        names (list): Image names
        labels (list): Image labels
        k_shot (int): Number of examples per class
        n_classes (int): Number of classes to sample
        seed (int): Random seed
        alpha (float): Parameter controlling focus on rare classes (0-1)
                      0 = uniform, 1 = focus entirely on rare classes
        
    Returns:
        tuple: (sampled_names, sampled_labels)
    """
    random.seed(seed)
    
    # Count examples per class
    class_counts = Counter(labels)
    
    # Calculate sampling weights (inverse frequency)
    total_examples = len(labels)
    class_weights = {cls: (total_examples / count) ** alpha for cls, count in class_counts.items()}
    
    # Normalize weights to probabilities
    weight_sum = sum(class_weights.values())
    class_probs = {cls: weight / weight_sum for cls, weight in class_weights.items()}
    
    # If n_classes is greater than available classes, use all available
    n_classes = min(n_classes, len(class_counts))
    
    # Sample n_classes according to weights
    classes = list(class_counts.keys())
    weights = [class_probs[cls] for cls in classes]
    selected_classes = random.choices(classes, weights=weights, k=n_classes)
    
    # Sample k examples from each selected class
    sampled_names = []
    sampled_labels = []
    
    for cls in selected_classes:
        # Get all examples of this class
        cls_indices = [i for i, label in enumerate(labels) if label == cls]
        
        # If we have fewer than k examples, use all available
        k = min(k_shot, len(cls_indices))
        
        # Randomly sample k examples
        sampled_indices = random.sample(cls_indices, k)
        
        # Add to our samples
        sampled_names.extend([names[i] for i in sampled_indices])
        sampled_labels.extend([labels[i] for i in sampled_indices])
    
    return sampled_names, sampled_labels

def generate_few_shot_splits(dataset_info, embeddings, names, labels, output_dir, 
                            k_shots=[1, 5, 10], n_classes=10, n_seeds=3):
    """
    Generate few-shot splits using different sampling strategies.
    
    Args:
        dataset_info (dict): Dataset information
        embeddings (numpy.ndarray): Image embeddings
        names (list): Image names
        labels (list): Image labels
        output_dir (str): Directory to save the splits
        k_shots (list): List of k values for k-shot sampling
        n_classes (int): Number of classes to sample
        n_seeds (int): Number of random seeds to use
    """
    dataset_name = dataset_info['name']
    splits_dir = os.path.join(output_dir, f"{dataset_name}_few_shot_splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    sampling_strategies = {
        'random': random_sampling,
        'stratified': stratified_sampling,
        'long_tail': long_tail_sampling
    }
    
    # Generate splits for each combination of parameters
    for k in k_shots:
        for strategy_name, strategy_func in sampling_strategies.items():
            for seed in range(n_seeds):
                # Generate split
                sampled_names, sampled_labels = strategy_func(
                    names, labels, k, n_classes, seed=seed
                )
                
                # Get embeddings for sampled images
                sampled_indices = [names.index(name) for name in sampled_names]
                sampled_embeddings = embeddings[sampled_indices]
                
                # Save split
                split_info = {
                    'dataset': dataset_name,
                    'strategy': strategy_name,
                    'k_shot': k,
                    'n_classes': n_classes,
                    'seed': seed,
                    'names': sampled_names,
                    'labels': sampled_labels
                }
                
                split_path = os.path.join(
                    splits_dir, 
                    f"{strategy_name}_{k}shot_{n_classes}classes_seed{seed}.json"
                )
                
                with open(split_path, 'w') as f:
                    json.dump(split_info, f, indent=2)
                
                # Save embeddings
                embeddings_path = os.path.join(
                    splits_dir, 
                    f"{strategy_name}_{k}shot_{n_classes}classes_seed{seed}_embeddings.npy"
                )
                
                np.save(embeddings_path, sampled_embeddings)
                
                logger.info(f"Saved {strategy_name} {k}-shot split with {len(sampled_names)} examples to {split_path}")

# ===============================
# Cross-dataset Generalization
# ===============================

def evaluate_cross_dataset_generalization(source_dataset, target_dataset, output_dir):
    """
    Evaluate cross-dataset generalization by measuring embedding space alignment.
    
    Args:
        source_dataset (dict): Source dataset information
        target_dataset (dict): Target dataset information
        output_dir (str): Directory to save the results
    """
    source_name = source_dataset['name']
    target_name = target_dataset['name']
    
    source_embeddings = source_dataset['embeddings']
    source_labels = source_dataset['labels']
    target_embeddings = target_dataset['embeddings']
    target_labels = target_dataset['labels']
    
    logger.info(f"Evaluating generalization from {source_name} to {target_name}")
    
    # Find common labels between datasets
    source_label_set = set(source_labels)
    target_label_set = set(target_labels)
    common_labels = source_label_set.intersection(target_label_set)
    
    if not common_labels:
        logger.warning(f"No common labels found between {source_name} and {target_name}")
        return
    
    logger.info(f"Found {len(common_labels)} common labels")
    
    # For each common label, compute centroid in source and target embedding space
    results = []
    
    for label in common_labels:
        # Get embeddings for this label in source dataset
        source_indices = [i for i, l in enumerate(source_labels) if l == label]
        if not source_indices:
            continue
        source_label_embeddings = source_embeddings[source_indices]
        source_centroid = np.mean(source_label_embeddings, axis=0)
        
        # Get embeddings for this label in target dataset
        target_indices = [i for i, l in enumerate(target_labels) if l == label]
        if not target_indices:
            continue
        target_label_embeddings = target_embeddings[target_indices]
        target_centroid = np.mean(target_label_embeddings, axis=0)
        
        # Compute cosine similarity between centroids
        centroid_similarity = np.dot(source_centroid, target_centroid)
        
        # Compute average distance from source centroid to target examples
        distances_to_target = np.dot(source_centroid, target_label_embeddings.T)
        avg_distance_to_target = np.mean(distances_to_target)
        
        # Compute average distance from target centroid to source examples
        distances_to_source = np.dot(target_centroid, source_label_embeddings.T)
        avg_distance_to_source = np.mean(distances_to_source)
        
        results.append({
            'label': label,
            'source_examples': len(source_indices),
            'target_examples': len(target_indices),
            'centroid_similarity': float(centroid_similarity),
            'avg_distance_source_to_target': float(avg_distance_to_target),
            'avg_distance_target_to_source': float(avg_distance_to_source)
        })
    
    # Save results
    results_path = os.path.join(output_dir, f"cross_dataset_{source_name}_to_{target_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(df['centroid_similarity'], df['avg_distance_source_to_target'], 
               alpha=0.7, s=df['source_examples'])
    
    for i, row in df.iterrows():
        plt.annotate(row['label'], 
                    (row['centroid_similarity'], row['avg_distance_source_to_target']),
                    fontsize=8)
    
    plt.xlabel('Centroid Similarity')
    plt.ylabel('Avg Distance (Source → Target)')
    plt.title(f'Cross-dataset Generalization: {source_name} → {target_name}')
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f"cross_dataset_{source_name}_to_{target_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Cross-dataset evaluation results saved to {results_path}")

# ===============================
# Visualization Utilities
# ===============================

def plot_duplicate_examples(duplicates, dataset_info, output_dir, max_examples=10, method='cosine', 
                           dataset2_info=None):
    """
    Plot examples of duplicate images.
    
    Args:
        duplicates (list): List of duplicate pairs
        dataset_info (dict): Dataset information for the first dataset
        output_dir (str): Directory to save the plots
        max_examples (int): Maximum number of examples to plot
        method (str): Method used for duplicate detection ('cosine' or 'perceptual')
        dataset2_info (dict, optional): Dataset information for the second dataset in cross-dataset duplicates
    """
    # Limit number of examples
    duplicates = duplicates[:max_examples]
    
    # Create figure
    n_examples = len(duplicates)
    fig, axes = plt.subplots(n_examples, 2, figsize=(10, n_examples * 3))
    
    # If only one example, make axes 2D
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    # Get image directories
    image_dir1 = os.path.dirname(dataset_info['image_paths'][0])
    image_dir2 = image_dir1 if dataset2_info is None else os.path.dirname(dataset2_info['image_paths'][0])
    
    # Get dataset names
    dataset1_name = dataset_info['name']
    dataset2_name = dataset1_name if dataset2_info is None else dataset2_info['name']
    
    # Plot each duplicate pair
    for i, (name1, name2, similarity) in enumerate(duplicates):
        # Load images
        img1_path = os.path.join(image_dir1, f"{name1}.jpg")
        img2_path = os.path.join(image_dir2, f"{name2}.jpg")
        
        try:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
            
            # Display images
            axes[i, 0].imshow(img1)
            axes[i, 1].imshow(img2)
            
            # Set titles
            if method == 'cosine':
                axes[i, 0].set_title(f"{dataset1_name}: {name1}")
                axes[i, 1].set_title(f"{dataset2_name}: {name2}\nSimilarity: {similarity:.3f}")
            else:  # perceptual
                axes[i, 0].set_title(f"{dataset1_name}: {name1}")
                axes[i, 1].set_title(f"{dataset2_name}: {name2}\nHash Difference: {similarity}")
            
            # Remove axis ticks
            axes[i, 0].set_xticks([])
            axes[i, 0].set_yticks([])
            axes[i, 1].set_xticks([])
            axes[i, 1].set_yticks([])
            
        except Exception as e:
            logger.error(f"Error plotting duplicate pair {name1} and {name2}: {e}")
            axes[i, 0].text(0.5, 0.5, f"Error loading {name1}", ha='center')
            axes[i, 1].text(0.5, 0.5, f"Error loading {name2}", ha='center')
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output filename
    if dataset2_info is None:
        # Within-dataset duplicates
        if method == 'cosine':
            output_path = os.path.join(output_dir, f"{dataset1_name}_duplicate_examples_cosine.png")
        else:
            output_path = os.path.join(output_dir, f"{dataset1_name}_duplicate_examples_perceptual.png")
    else:
        # Cross-dataset duplicates
        if method == 'cosine':
            output_path = os.path.join(output_dir, f"{dataset1_name}_{dataset2_name}_duplicate_examples_cosine.png")
        else:
            output_path = os.path.join(output_dir, f"{dataset1_name}_{dataset2_name}_duplicate_examples_perceptual.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Duplicate examples plot saved to {output_path}")

def plot_venn_diagram(datasets, duplicates_info, output_dir, method='cosine'):
    """
    Plot Venn diagram showing overlap between datasets.
    
    Args:
        datasets (list): List of dataset information dictionaries
        duplicates_info (dict): Dictionary with duplicate counts
        output_dir (str): Directory to save the plot
        method (str): Method used for duplicate detection ('cosine' or 'perceptual')
    """
    # Extract dataset names and sizes
    dataset_names = [dataset['name'] for dataset in datasets]
    dataset_sizes = [len(dataset['names']) for dataset in datasets]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    if len(datasets) == 2:
        # Two-dataset case
        set1_size = dataset_sizes[0]
        set2_size = dataset_sizes[1]
        
        # Get overlap count - number of images from dataset1 that have a similar image in dataset2
        overlap_key = f"{dataset_names[0]}_{dataset_names[1]}"
        overlap_count = duplicates_info.get(overlap_key, 0)
        
        # Ensure overlap doesn't exceed dataset sizes
        overlap_count = min(overlap_count, set1_size, set2_size)
        
        # Create Venn diagram with proper proportions
        venn2(subsets=(set1_size - overlap_count, set2_size - overlap_count, overlap_count),
              set_labels=(f"{dataset_names[0]} ({set1_size})", f"{dataset_names[1]} ({set2_size})"))
        
    elif len(datasets) == 3:
        # Three-dataset case
        set1_size = dataset_sizes[0]
        set2_size = dataset_sizes[1]
        set3_size = dataset_sizes[2]
        
        # Get pairwise overlap counts
        overlap_12_key = f"{dataset_names[0]}_{dataset_names[1]}"
        overlap_13_key = f"{dataset_names[0]}_{dataset_names[2]}"
        overlap_23_key = f"{dataset_names[1]}_{dataset_names[2]}"
        
        overlap_12 = duplicates_info.get(overlap_12_key, 0)
        overlap_13 = duplicates_info.get(overlap_13_key, 0)
        overlap_23 = duplicates_info.get(overlap_23_key, 0)
        
        # Ensure pairwise overlaps don't exceed dataset sizes
        overlap_12 = min(overlap_12, set1_size, set2_size)
        overlap_13 = min(overlap_13, set1_size, set3_size)
        overlap_23 = min(overlap_23, set2_size, set3_size)
        
        # Estimate triple overlap - images that are similar across all three datasets
        # This is a more accurate estimation based on actual duplicate counts
        overlap_123 = min(
            # Count of dataset1 images with similar images in both dataset2 and dataset3
            sum(1 for name1, _, _ in duplicates_info.get(f"{dataset_names[0]}_{dataset_names[1]}_duplicates", [])
                if any(name1 == name1b for name1b, _, _ in duplicates_info.get(f"{dataset_names[0]}_{dataset_names[2]}_duplicates", []))),
            
            # Count of dataset2 images with similar images in both dataset1 and dataset3
            sum(1 for _, name2, _ in duplicates_info.get(f"{dataset_names[0]}_{dataset_names[1]}_duplicates", [])
                if any(name2 == name2b for name2b, _, _ in duplicates_info.get(f"{dataset_names[1]}_{dataset_names[2]}_duplicates", []))),
            
            # Count of dataset3 images with similar images in both dataset1 and dataset2
            sum(1 for _, name3, _ in duplicates_info.get(f"{dataset_names[0]}_{dataset_names[2]}_duplicates", [])
                if any(name3 == name3b for _, name3b, _ in duplicates_info.get(f"{dataset_names[1]}_{dataset_names[2]}_duplicates", [])))
        )
        
        # If we don't have the actual duplicate lists, fall back to a simpler estimation
        if overlap_123 == 0:
            overlap_123 = min(overlap_12, overlap_13, overlap_23) // 3
        
        # Adjust pairwise overlaps to account for triple overlap
        overlap_12_only = overlap_12 - overlap_123
        overlap_13_only = overlap_13 - overlap_123
        overlap_23_only = overlap_23 - overlap_123
        
        # Calculate exclusive regions
        set1_only = max(0, set1_size - overlap_12_only - overlap_13_only - overlap_123)
        set2_only = max(0, set2_size - overlap_12_only - overlap_23_only - overlap_123)
        set3_only = max(0, set3_size - overlap_13_only - overlap_23_only - overlap_123)
        
        # Create Venn diagram with proper proportions
        venn3(subsets=(set1_only, set2_only, overlap_12_only, 
                       set3_only, overlap_13_only, overlap_23_only, overlap_123),
              set_labels=(f"{dataset_names[0]} ({set1_size})", 
                          f"{dataset_names[1]} ({set2_size})", 
                          f"{dataset_names[2]} ({set3_size})"))
    else:
        logger.warning(f"Venn diagram not supported for {len(datasets)} datasets")
        return
    
    # Set title
    if method == 'cosine':
        plt.title(f"Dataset Overlap using Cosine Similarity")
    else:
        plt.title(f"Dataset Overlap using Perceptual Hashing")
    
    # Save figure
    output_path = os.path.join(output_dir, f"venn_diagram_{method}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Venn diagram saved to {output_path}")

# ===============================
# Main Function
# ===============================

@click.command()
@click.option('--datasets', multiple=True, default=['ArtDL', 'ICONCLASS', 'wikidata'],
              help='List of datasets to analyze')
@click.option('--max-images', type=int, default=None,
              help='Maximum number of images to process per dataset')
@click.option('--output-dir', default='overlap_analysis',
              help='Directory to save results')
@click.option('--clip-model', default='ViT-B/32',
              help='CLIP model to use (ViT-B/32, ViT-B/16, ViT-L/14)')
@click.option('--batch-size', type=int, default=32,
              help='Batch size for processing images')
@click.option('--method', type=click.Choice(['cosine', 'perceptual', 'both']), default='both',
              help='Method for duplicate detection')
@click.option('--cosine-threshold', type=float, default=0.9,
              help='Similarity threshold for cosine-based duplicate detection')
@click.option('--hash-threshold', type=int, default=5,
              help='Difference threshold for perceptual hash-based duplicate detection')
@click.option('--hash-size', type=int, default=8,
              help='Size of perceptual hash')
@click.option('--hash-type', type=click.Choice(['phash', 'dhash', 'whash', 'ahash']), default='phash',
              help='Type of perceptual hash')
@click.option('--projections', type=click.Choice(['tsne', 'umap', 'both']), default='both',
              help='Projection methods to use')
@click.option('--tsne-perplexity', type=int, default=30,
              help='Perplexity parameter for t-SNE')
@click.option('--umap-neighbors', type=int, default=15,
              help='Number of neighbors for UMAP')
@click.option('--use-ground-truth', is_flag=True,
              help='Use ground truth labels from 2_ground_truth.json for clustering')
@click.option('--generate-few-shot', is_flag=True,
              help='Generate few-shot splits')
@click.option('--k-shots', multiple=True, type=int, default=[1, 5, 10],
              help='Number of examples per class for few-shot sampling')
@click.option('--n-classes', type=int, default=10,
              help='Number of classes for few-shot sampling')
@click.option('--n-seeds', type=int, default=3,
              help='Number of random seeds for few-shot sampling')
@click.option('--evaluate-cross-dataset', is_flag=True,
              help='Evaluate cross-dataset generalization')
@click.option('--verbose', is_flag=True, help='Enable verbose logging (DEBUG level)')
def main(datasets, max_images, output_dir, clip_model, batch_size, method, 
         cosine_threshold, hash_threshold, hash_size, hash_type, projections,
         tsne_perplexity, umap_neighbors, use_ground_truth, generate_few_shot, k_shots, n_classes,
         n_seeds, evaluate_cross_dataset, verbose):
    """
    Analyze dataset overlap, detect duplicates, and prepare benchmarks.
    
    This script provides functionality for:
    1. Extracting CLIP embeddings from test datasets
    2. Detecting duplicates using perceptual hashing or cosine similarity
    3. Projecting embeddings using t-SNE and UMAP
    4. Preparing few-shot benchmarks with various sampling strategies
    5. Evaluating cross-dataset generalization
    """
    # Set up logging level based on verbose flag
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    else:
        logger.setLevel(logging.INFO)
    
    # Create base directories
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    
    # Create cross-dataset analysis directory
    cross_dir = os.path.join(base_dir, 'dataset', 'cross')
    os.makedirs(cross_dir, exist_ok=True)
    
    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load CLIP model
    logger.info(f"Loading CLIP model: {clip_model}")
    model, preprocess = clip.load(clip_model, device=device)
    
    # Process each dataset
    all_datasets = []
    
    for dataset_name in datasets:
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Load dataset information
        dataset_info = load_dataset_info(dataset_name, base_dir, use_ground_truth)
        analysis_dir = dataset_info['analysis_dir']
        
        # Load images
        images = load_images(dataset_info, max_images)
        logger.info(f"Loaded {len(images)} images from {dataset_name}")
        
        # Extract CLIP embeddings
        names, embeddings, labels = extract_clip_embeddings(
            images, model, preprocess, device, batch_size
        )
        
        # Save embeddings to dataset-specific analysis directory
        np.save(os.path.join(analysis_dir, f'embeddings.npy'), embeddings)
        with open(os.path.join(analysis_dir, f'names.json'), 'w') as f:
            json.dump(names, f)
        with open(os.path.join(analysis_dir, f'labels.json'), 'w') as f:
            json.dump(labels, f)
        
        logger.info(f"Saved embeddings for {len(names)} images to {analysis_dir}")
        
        # Compute perceptual hashes if needed
        if method in ['perceptual', 'both']:
            hash_names, hashes, hash_labels = compute_perceptual_hashes(
                images, hash_size, hash_type
            )
            
            # Store hashes (convert to strings for JSON serialization)
            hash_strings = [str(h) for h in hashes]
            with open(os.path.join(analysis_dir, f'hashes.json'), 'w') as f:
                json.dump(hash_strings, f)
            
            logger.info(f"Computed {hash_type} hashes for {len(hash_names)} images")
        else:
            hashes = None
        
        # Store dataset information
        all_datasets.append({
            'name': dataset_name,
            'info': dataset_info,
            'names': names,
            'embeddings': embeddings,
            'labels': labels,
            'hashes': hashes
        })
        
        # Project embeddings for individual dataset
        if projections in ['tsne', 'both']:
            tsne_df = project_embeddings_tsne(
                embeddings, names, labels, 
                perplexity=tsne_perplexity
            )
            # Add label source to title
            label_source = "Ground Truth Labels" if use_ground_truth else "Default Labels"
            plot_projection(
                tsne_df,
                f't-SNE Projection of {dataset_name} ({label_source})',
                os.path.join(analysis_dir, f'tsne.html'),
                projection_type='t-SNE'
            )
        
        if projections in ['umap', 'both']:
            umap_df = project_embeddings_umap(
                embeddings, names, labels,
                n_neighbors=umap_neighbors
            )
            # Add label source to title
            label_source = "Ground Truth Labels" if use_ground_truth else "Default Labels"
            plot_projection(
                umap_df,
                f'UMAP Projection of {dataset_name} ({label_source})',
                os.path.join(analysis_dir, f'umap.html'),
                projection_type='UMAP'
            )
        
        # Generate few-shot splits if requested
        if generate_few_shot:
            generate_few_shot_splits(
                dataset_info, embeddings, names, labels, analysis_dir,
                k_shots=k_shots, n_classes=n_classes, n_seeds=n_seeds
            )
    
    # Detect duplicates within each dataset
    for dataset in all_datasets:
        dataset_name = dataset['name']
        analysis_dir = dataset['info']['analysis_dir']
        
        # Cosine similarity-based duplicates
        if method in ['cosine', 'both']:
            cosine_duplicates = detect_duplicates_cosine(
                dataset['embeddings'], dataset['names'], cosine_threshold
            )
            
            # Save duplicates
            with open(os.path.join(analysis_dir, f'cosine_duplicates.json'), 'w') as f:
                json.dump(cosine_duplicates, f, indent=2)
            
            logger.info(f"Found {len(cosine_duplicates)} cosine-based duplicates in {dataset_name}")
            
            # Plot examples
            if cosine_duplicates:
                plot_duplicate_examples(
                    cosine_duplicates, dataset['info'], analysis_dir, method='cosine'
                )
        
        # Perceptual hash-based duplicates
        if method in ['perceptual', 'both'] and dataset['hashes'] is not None:
            perceptual_duplicates = detect_duplicates_perceptual(
                dataset['hashes'], dataset['names'], hash_threshold
            )
            
            # Save duplicates
            with open(os.path.join(analysis_dir, f'perceptual_duplicates.json'), 'w') as f:
                json.dump(perceptual_duplicates, f, indent=2)
            
            logger.info(f"Found {len(perceptual_duplicates)} perceptual hash-based duplicates in {dataset_name}")
            
            # Plot examples
            if perceptual_duplicates:
                plot_duplicate_examples(
                    perceptual_duplicates, dataset['info'], analysis_dir, method='perceptual'
                )
    
    # Detect cross-dataset duplicates
    cosine_duplicates_info = {}
    perceptual_duplicates_info = {}
    
    # Store actual duplicate lists for better triple overlap estimation
    cosine_duplicates_lists = {}
    perceptual_duplicates_lists = {}
    
    for i, dataset1 in enumerate(all_datasets):
        for j, dataset2 in enumerate(all_datasets):
            if i >= j:  # Skip self-comparisons and duplicates
                continue
            
            dataset1_name = dataset1['name']
            dataset2_name = dataset2['name']
            
            # Cosine similarity-based duplicates
            if method in ['cosine', 'both']:
                cross_cosine_duplicates = detect_cross_dataset_duplicates(
                    dataset1['embeddings'], dataset1['names'],
                    dataset2['embeddings'], dataset2['names'],
                    cosine_threshold
                )
                
                # Save duplicates
                with open(os.path.join(cross_dir, f'{dataset1_name}_{dataset2_name}_cosine_duplicates.json'), 'w') as f:
                    json.dump(cross_cosine_duplicates, f, indent=2)
                
                logger.info(f"Found {len(cross_cosine_duplicates)} cosine-based duplicates between {dataset1_name} and {dataset2_name}")
                
                # Plot examples
                if cross_cosine_duplicates:
                    plot_duplicate_examples(
                        cross_cosine_duplicates, dataset1['info'], cross_dir, method='cosine',
                        dataset2_info=dataset2['info']
                    )
                
                # Store count and list for Venn diagram
                cosine_duplicates_info[f"{dataset1_name}_{dataset2_name}"] = len(cross_cosine_duplicates)
                cosine_duplicates_lists[f"{dataset1_name}_{dataset2_name}_duplicates"] = cross_cosine_duplicates
            
            # Perceptual hash-based duplicates
            if method in ['perceptual', 'both'] and dataset1['hashes'] is not None and dataset2['hashes'] is not None:
                cross_perceptual_duplicates = detect_cross_dataset_duplicates_perceptual(
                    dataset1['hashes'], dataset1['names'],
                    dataset2['hashes'], dataset2['names'],
                    hash_threshold
                )
                
                # Save duplicates
                with open(os.path.join(cross_dir, f'{dataset1_name}_{dataset2_name}_perceptual_duplicates.json'), 'w') as f:
                    json.dump(cross_perceptual_duplicates, f, indent=2)
                
                logger.info(f"Found {len(cross_perceptual_duplicates)} perceptual hash-based duplicates between {dataset1_name} and {dataset2_name}")
                
                # Plot examples
                if cross_perceptual_duplicates:
                    plot_duplicate_examples(
                        cross_perceptual_duplicates, dataset1['info'], cross_dir, method='perceptual',
                        dataset2_info=dataset2['info']
                    )
                
                # Store count and list for Venn diagram
                perceptual_duplicates_info[f"{dataset1_name}_{dataset2_name}"] = len(cross_perceptual_duplicates)
                perceptual_duplicates_lists[f"{dataset1_name}_{dataset2_name}_duplicates"] = cross_perceptual_duplicates
    
    # Plot combined projections
    if len(all_datasets) > 1:
        if projections in ['tsne', 'both']:
            plot_combined_projection(all_datasets, 'tsne', cross_dir)
        
        if projections in ['umap', 'both']:
            plot_combined_projection(all_datasets, 'umap', cross_dir)
        
        # Plot Venn diagrams
        if method in ['cosine', 'both']:
            # Merge duplicates info with lists for better triple overlap estimation
            cosine_info_with_lists = {**cosine_duplicates_info, **cosine_duplicates_lists}
            plot_venn_diagram(all_datasets, cosine_info_with_lists, cross_dir, method='cosine')
        
        if method in ['perceptual', 'both']:
            # Merge duplicates info with lists for better triple overlap estimation
            perceptual_info_with_lists = {**perceptual_duplicates_info, **perceptual_duplicates_lists}
            plot_venn_diagram(all_datasets, perceptual_info_with_lists, cross_dir, method='perceptual')
    
    # Evaluate cross-dataset generalization if requested
    if evaluate_cross_dataset and len(all_datasets) > 1:
        for i, source_dataset in enumerate(all_datasets):
            for j, target_dataset in enumerate(all_datasets):
                if i != j:  # Skip self-evaluation
                    evaluate_cross_dataset_generalization(
                        source_dataset, target_dataset, cross_dir
                    )
    
    logger.info(f"Analysis complete. Results saved to dataset-specific analysis directories and {cross_dir}")

if __name__ == '__main__':
    main()
