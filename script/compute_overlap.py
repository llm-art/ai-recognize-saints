#!/usr/bin/env python3
"""
Trimmed script for analyzing image dataset overlap using perceptual hashing.

This script provides focused functionality for:
1. Loading images from datasets
2. Computing hashes for all images
3. Calculating hamming distance between images of different datasets only
4. Generating Venn diagram visualizations of cross-dataset similarity
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib_venn import venn3, venn3_circles
from PIL import Image
import imagehash
from tqdm import tqdm
import json
import logging
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
    
    # Create analysis directory in the new location
    analysis_dir = os.path.join(base_dir, 'dataset', 'analysis', dataset_name)
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
# Perceptual Hashing Functions
# ===============================

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
# Robust Hashing Functions
# ===============================

def block_mean_hash(image, hash_size=16):
    """
    Generates a robust hash using block mean value approach with optimizations.
    
    Args:
        image (PIL.Image): Input image
        hash_size (int): Size of the hash
        
    Returns:
        str: Hash string
    """
    # Convert to grayscale if needed
    if image.mode != "L":
        image = image.convert("L")

    # Normalize image size
    norm_size = (hash_size, hash_size)
    image = image.resize(norm_size, Image.LANCZOS)  # LANCZOS is the new name for ANTIALIAS

    # Convert image to numpy array
    pixels = np.array(image)
    
    # Divide into 4 subareas (2x2)
    h, w = pixels.shape
    h2, w2 = h // 2, w // 2
    subareas = [
        pixels[:h2, :w2],
        pixels[:h2, w2:],
        pixels[h2:, :w2],
        pixels[h2:, w2:]
    ]

    # Calculate mean of each subarea
    means = [np.mean(sa) for sa in subareas]

    # Find index of darkest subarea (for mirroring consistency)
    darkest_index = np.argmin(means)

    # Mirror image if darkest is not at top-left
    if darkest_index != 0:
        pixels = np.fliplr(pixels)

    # Recompute full normalized image with updated pixels
    mean_val = np.mean(pixels)
    hash_bits = (pixels >= mean_val).astype(int).flatten()

    return ''.join(str(b) for b in hash_bits)

def compute_robust_hashes(images, hash_size=16):
    """
    Compute robust hashes for images.
    
    Args:
        images (list): List of tuples (image_name, PIL.Image, label)
        hash_size (int): Size of the hash
        
    Returns:
        tuple: (image_names, hashes, labels)
    """
    all_names = []
    all_hashes = []
    all_labels = []
    
    for name, image, label in tqdm(images, desc="Computing robust hashes"):
        try:
            img_hash = block_mean_hash(image, hash_size=hash_size)
            all_names.append(name)
            all_hashes.append(img_hash)
            all_labels.append(label)
        except Exception as e:
            logger.error(f"Error computing robust hash for image {name}: {e}")
    
    return all_names, all_hashes, all_labels

def hamming_distance(hash1, hash2):
    """
    Compute the Hamming distance between two hash strings using optimized methods.
    
    This implementation uses:
    1. Bitwise operations for binary strings (faster for shorter strings)
    2. NumPy for array-based operations (faster for longer strings)
    3. Appropriate fallbacks for non-binary strings
    
    Args:
        hash1 (str): First hash string
        hash2 (str): Second hash string
        
    Returns:
        int: Hamming distance
    """
    # Handle different length strings
    if len(hash1) != len(hash2):
        logger.warning(f"Hash strings have different lengths: {len(hash1)} vs {len(hash2)}")
        min_len = min(len(hash1), len(hash2))
        length_diff = abs(len(hash1) - len(hash2))
        
        # For shorter strings (< 64 bits), use bitwise operations
        if min_len <= 64:
            try:
                # Try to convert the common parts to integers and use XOR
                int1 = int(hash1[:min_len], 2)
                int2 = int(hash2[:min_len], 2)
                xor_result = int1 ^ int2
                return bin(xor_result).count('1') + length_diff
            except ValueError:
                # If not binary strings, fall back to character comparison
                return sum(hash1[i] != hash2[i] for i in range(min_len)) + length_diff
        else:
            # For longer strings, use NumPy
            try:
                # Convert to numpy arrays of integers
                arr1 = np.array([int(c) for c in hash1[:min_len]])
                arr2 = np.array([int(c) for c in hash2[:min_len]])
                return np.sum(arr1 != arr2) + length_diff
            except ValueError:
                # If conversion fails, fall back to character comparison
                return sum(hash1[i] != hash2[i] for i in range(min_len)) + length_diff
    
    # For equal length strings
    # For shorter strings (< 64 bits), use bitwise operations
    if len(hash1) <= 64:
        try:
            # Try to convert to integers and use XOR
            int1 = int(hash1, 2)
            int2 = int(hash2, 2)
            xor_result = int1 ^ int2
            return bin(xor_result).count('1')
        except ValueError:
            # If not binary strings, fall back to character comparison
            return sum(ch1 != ch2 for ch1, ch2 in zip(hash1, hash2))
    else:
        # For longer strings, use NumPy
        try:
            # Convert to numpy arrays of integers
            arr1 = np.array([int(c) for c in hash1])
            arr2 = np.array([int(c) for c in hash2])
            return np.sum(arr1 != arr2)
        except ValueError:
            # If conversion fails, fall back to character comparison
            return sum(ch1 != ch2 for ch1, ch2 in zip(hash1, hash2))

# ===============================
# Cross-Dataset Duplicate Detection
# ===============================

def detect_cross_dataset_duplicates(hashes1, names1, hashes2, names2, threshold=5, method='perceptual'):
    """
    Detect duplicate images across two datasets using hash similarity.
    
    Args:
        hashes1 (list): Hashes from first dataset
        names1 (list): Image names from first dataset
        hashes2 (list): Hashes from second dataset
        names2 (list): Image names from second dataset
        threshold (int): Maximum hash difference for considering duplicates
        method (str): Method used for similarity calculation ('perceptual' or 'robust')
        
    Returns:
        list: List of duplicate pairs (name1, name2, difference)
    """
    duplicates = []
    
    # Log the number of comparisons that will be made
    total_comparisons = len(names1) * len(names2)
    logger.info(f"Comparing {len(names1)} images from dataset 1 with {len(names2)} images from dataset 2 ({total_comparisons} comparisons)")
    
    # Use tqdm to show progress
    for i in tqdm(range(len(names1)), desc=f"{method.capitalize()} hash comparison"):
        try:
            hash1 = hashes1[i]
            
            # Process hashes in batches for better performance
            batch_size = 100
            for batch_start in range(0, len(names2), batch_size):
                batch_end = min(batch_start + batch_size, len(names2))
                
                for j in range(batch_start, batch_end):
                    try:
                        # Calculate difference based on hash type
                        if method == 'perceptual':
                            # For perceptual hashes (imagehash objects), use the - operator
                            difference = hash1 - hashes2[j]
                        else:
                            # For robust hashes (strings), use the hamming_distance function
                            difference = hamming_distance(hash1, hashes2[j])
                            
                        if difference <= threshold:
                            duplicates.append((names1[i], names2[j], int(difference)))
                    except Exception as e:
                        logger.error(f"Error comparing hashes for {names1[i]} and {names2[j]}: {e}")
        except Exception as e:
            logger.error(f"Error processing hash at index {i}: {e}")
    
    return duplicates

# ===============================
# Venn Diagram Visualization
# ===============================

def generate_venn_diagram(all_datasets, method='perceptual', threshold=8, output_path=None):
    """
    Generate a Venn diagram showing the overlap between datasets.
    
    Args:
        all_datasets (list): List of dataset information dictionaries
        method (str): Method used for similarity calculation ('perceptual' or 'robust')
        threshold (int): Threshold for considering images as similar
        output_path (str): Path to save the Venn diagram
        
    Returns:
        None
    """
    if len(all_datasets) != 3:
        logger.warning(f"Venn diagram requires exactly 3 datasets, but {len(all_datasets)} were provided.")
        return
    
    # Extract dataset names and sizes
    dataset_names = [d['name'] for d in all_datasets]
    dataset_sizes = [len(d[f'{method}_names']) for d in all_datasets]
    
    # Find duplicates between each pair of datasets
    duplicates_01 = set()
    duplicates_02 = set()
    duplicates_12 = set()
    
    # Dataset 0 vs Dataset 1
    dups_01 = detect_cross_dataset_duplicates(
        all_datasets[0][f'{method}_hashes'], all_datasets[0][f'{method}_names'],
        all_datasets[1][f'{method}_hashes'], all_datasets[1][f'{method}_names'],
        threshold, method
    )
    
    for name1, name2, _ in dups_01:
        duplicates_01.add((name1, name2))
    
    # Dataset 0 vs Dataset 2
    dups_02 = detect_cross_dataset_duplicates(
        all_datasets[0][f'{method}_hashes'], all_datasets[0][f'{method}_names'],
        all_datasets[2][f'{method}_hashes'], all_datasets[2][f'{method}_names'],
        threshold, method
    )
    
    for name1, name2, _ in dups_02:
        duplicates_02.add((name1, name2))
    
    # Dataset 1 vs Dataset 2
    dups_12 = detect_cross_dataset_duplicates(
        all_datasets[1][f'{method}_hashes'], all_datasets[1][f'{method}_names'],
        all_datasets[2][f'{method}_hashes'], all_datasets[2][f'{method}_names'],
        threshold, method
    )
    
    for name1, name2, _ in dups_12:
        duplicates_12.add((name1, name2))
    
    # Find images that are similar across all three datasets
    # Create dictionaries to map from dataset 0 to datasets 1 and 2
    matches_0_to_1 = {pair[0]: pair[1] for pair in duplicates_01}
    matches_0_to_2 = {pair[0]: pair[1] for pair in duplicates_02}
    
    # Create a set of matches between datasets 1 and 2
    matches_1_to_2 = {pair[0]: pair[1] for pair in duplicates_12}
    
    # Find images in dataset 0 that match with both datasets 1 and 2
    common_in_0 = set(matches_0_to_1.keys()) & set(matches_0_to_2.keys())
    
    # Check if the corresponding matches in datasets 1 and 2 also match with each other
    triple_matches = 0
    for img_0 in common_in_0:
        img_1 = matches_0_to_1[img_0]
        img_2 = matches_0_to_2[img_0]
        
        # Check if img_1 matches with img_2
        if img_1 in matches_1_to_2 and matches_1_to_2[img_1] == img_2:
            triple_matches += 1
    
    # Calculate unique overlaps (excluding triple overlap)
    ab_only = len(duplicates_01) - triple_matches
    ac_only = len(duplicates_02) - triple_matches
    bc_only = len(duplicates_12) - triple_matches
    
    # Calculate unique elements in each dataset
    a_only = dataset_sizes[0] - ab_only - ac_only - triple_matches
    b_only = dataset_sizes[1] - ab_only - bc_only - triple_matches
    c_only = dataset_sizes[2] - ac_only - bc_only - triple_matches
    
    # Create the Venn diagram
    plt.figure(figsize=(10, 10))
    
    # Create a list of set sizes in the order expected by venn3
    # Order: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    # Where uppercase letters indicate inclusion in the set
    set_sizes = (a_only, b_only, ab_only, c_only, ac_only, bc_only, triple_matches)
    
    # Create the Venn diagram
    venn = venn3(subsets=set_sizes, set_labels=dataset_names)
    
    # Add title
    plt.title(f'Dataset Overlap ({method.capitalize()} Hash Similarity)', fontsize=18)
    
    # Add a legend with dataset sizes
    legend_labels = [f"{name} ({size} images)" for name, size in zip(dataset_names, dataset_sizes)]
    plt.figlegend(handles=venn.patches, labels=legend_labels, loc='upper center', 
                 bbox_to_anchor=(0.5, 0), ncol=3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Venn diagram saved to {output_path}")
    
    plt.close()

# ===============================
# Main Function
# ===============================

@click.command()
@click.option('--datasets', multiple=True, default=['ArtDL', 'ICONCLASS', 'wikidata'],
              help='List of datasets to analyze')
@click.option('--max-images', type=int, default=None,
              help='Maximum number of images to process per dataset')
@click.option('--output-dir', default='dataset/analysis',
              help='Directory to save results')
@click.option('--perceptual-hash-size', type=int, default=8,
              help='Size of perceptual hash')
@click.option('--robust-hash-size', type=int, default=16,
              help='Size of robust hash')
@click.option('--perceptual-hash-type', type=click.Choice(['phash', 'dhash', 'whash', 'ahash']), default='phash',
              help='Type of perceptual hash')
@click.option('--perceptual-threshold', type=int, default=8,
              help='Difference threshold for perceptual hash-based duplicate detection')
@click.option('--robust-threshold', type=int, default=8,
              help='Difference threshold for robust hash-based duplicate detection')
@click.option('--use-ground-truth', is_flag=True,
              help='Use ground truth labels from 2_ground_truth.json')
@click.option('--verbose', is_flag=True, help='Enable verbose logging (DEBUG level)')
def main(datasets, max_images, output_dir, perceptual_hash_size, robust_hash_size, 
         perceptual_hash_type, perceptual_threshold, robust_threshold, 
         use_ground_truth, verbose):
    """
    Analyze cross-dataset overlap using perceptual and robust hashing.
    
    This script provides focused functionality for:
    1. Loading images from datasets
    2. Computing hashes for all images
    3. Calculating hamming distance between images of different datasets only
    4. Generating Venn diagram visualizations of cross-dataset similarity
    """
    # Set up logging level based on verbose flag
    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    else:
        logger.setLevel(logging.INFO)
    
    # Create base directories
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        # Compute perceptual hashes
        perceptual_names, perceptual_hashes, perceptual_labels = compute_perceptual_hashes(
            images, hash_size=perceptual_hash_size, hash_type=perceptual_hash_type
        )
        logger.info(f"Computed {perceptual_hash_type} hashes for {len(perceptual_names)} images")
        
        # Compute robust hashes
        robust_names, robust_hashes, robust_labels = compute_robust_hashes(
            images, hash_size=robust_hash_size
        )
        logger.info(f"Computed robust hashes for {len(robust_names)} images")
        
        # Store hash strings for JSON serialization
        perceptual_hash_strings = [str(h) for h in perceptual_hashes]
        
        # Save hashes
        with open(os.path.join(analysis_dir, f'perceptual_hashes.json'), 'w') as f:
            json.dump(perceptual_hash_strings, f)
        
        with open(os.path.join(analysis_dir, f'robust_hashes.json'), 'w') as f:
            json.dump(robust_hashes, f)
        
        # Store dataset information
        all_datasets.append({
            'name': dataset_name,
            'info': dataset_info,
            'perceptual_names': perceptual_names,
            'perceptual_hashes': perceptual_hashes,
            'robust_names': robust_names,
            'robust_hashes': robust_hashes
        })
    
    # Detect cross-dataset duplicates
    if len(all_datasets) > 1:
        # Create main analysis directory for cross-dataset results
        main_analysis_dir = os.path.join(base_dir, output_dir)
        os.makedirs(main_analysis_dir, exist_ok=True)
        
        # Lists to store all cross-dataset duplicates
        all_perceptual_duplicates = []
        all_robust_duplicates = []
        
        # Process each dataset pair
        for i, dataset1 in enumerate(all_datasets):
            for j, dataset2 in enumerate(all_datasets):
                if i >= j:  # Skip self-comparisons and duplicates
                    continue
                
                dataset1_name = dataset1['name']
                dataset2_name = dataset2['name']
                
                # Detect cross-dataset duplicates using perceptual hashing
                perceptual_duplicates = detect_cross_dataset_duplicates(
                    dataset1['perceptual_hashes'], dataset1['perceptual_names'],
                    dataset2['perceptual_hashes'], dataset2['perceptual_names'],
                    perceptual_threshold, 'perceptual'
                )
                
                logger.info(f"Found {len(perceptual_duplicates)} perceptual hash-based duplicates between {dataset1_name} and {dataset2_name}")
                
                # Add to all perceptual duplicates
                for name1, name2, difference in perceptual_duplicates:
                    duplicate_tuple = [
                        {
                            "dataset": dataset1_name,
                            "name": name1,
                            "path": os.path.join(os.path.dirname(dataset1['info']['image_paths'][0]), f"{name1}.jpg")
                        },
                        {
                            "dataset": dataset2_name,
                            "name": name2,
                            "path": os.path.join(os.path.dirname(dataset2['info']['image_paths'][0]), f"{name2}.jpg")
                        }
                    ]
                    all_perceptual_duplicates.append(duplicate_tuple)
                
                # Detect cross-dataset duplicates using robust hashing
                robust_duplicates = detect_cross_dataset_duplicates(
                    dataset1['robust_hashes'], dataset1['robust_names'],
                    dataset2['robust_hashes'], dataset2['robust_names'],
                    robust_threshold, 'robust'
                )
                
                logger.info(f"Found {len(robust_duplicates)} robust hash-based duplicates between {dataset1_name} and {dataset2_name}")
                
                # Add to all robust duplicates
                for name1, name2, difference in robust_duplicates:
                    duplicate_tuple = [
                        {
                            "dataset": dataset1_name,
                            "name": name1,
                            "path": os.path.join(os.path.dirname(dataset1['info']['image_paths'][0]), f"{name1}.jpg")
                        },
                        {
                            "dataset": dataset2_name,
                            "name": name2,
                            "path": os.path.join(os.path.dirname(dataset2['info']['image_paths'][0]), f"{name2}.jpg")
                        }
                    ]
                    all_robust_duplicates.append(duplicate_tuple)
        
        # Save consolidated perceptual duplicates
        perceptual_duplicates_path = os.path.join(main_analysis_dir, 'perceptual_cross_duplicates.json')
        with open(perceptual_duplicates_path, 'w') as f:
            json.dump(all_perceptual_duplicates, f, indent=2)
        
        logger.info(f"Saved {len(all_perceptual_duplicates)} consolidated perceptual cross-dataset duplicates to {perceptual_duplicates_path}")
        
        # Save consolidated robust duplicates
        robust_duplicates_path = os.path.join(main_analysis_dir, 'robust_cross_duplicates.json')
        with open(robust_duplicates_path, 'w') as f:
            json.dump(all_robust_duplicates, f, indent=2)
        
        logger.info(f"Saved {len(all_robust_duplicates)} consolidated robust cross-dataset duplicates to {robust_duplicates_path}")
        
        # Generate Venn diagrams if we have exactly 3 datasets
        if len(all_datasets) == 3:
            # Generate perceptual hash Venn diagram
            perceptual_venn_path = os.path.join(main_analysis_dir, 'perceptual_venn_diagram.png')
            generate_venn_diagram(
                all_datasets, method='perceptual', 
                threshold=perceptual_threshold, output_path=perceptual_venn_path
            )
            
            # Generate robust hash Venn diagram
            robust_venn_path = os.path.join(main_analysis_dir, 'robust_venn_diagram.png')
            generate_venn_diagram(
                all_datasets, method='robust', 
                threshold=robust_threshold, output_path=robust_venn_path
            )
        else:
            logger.warning(f"Venn diagram requires exactly 3 datasets, but {len(all_datasets)} were provided.")
        
        # Create README.md
        readme_path = os.path.join(main_analysis_dir, 'README.md')
        
        # Create datasets list
        datasets_list = ""
        for dataset in all_datasets:
            datasets_list += f"- {dataset['name']} ({len(dataset['perceptual_names'])} images)\n"
        
        # Create dataset files list
        dataset_files_list = ""
        for dataset in all_datasets:
            dataset_files_list += f"- **{dataset['name']}**:\n"
            dataset_files_list += f"  - Perceptual hashes: `{dataset['name']}/perceptual_hashes.json`\n"
            dataset_files_list += f"  - Robust hashes: `{dataset['name']}/robust_hashes.json`\n"
        
        # Create detailed results for the README
        
        # 1. Overall duplicate counts
        total_perceptual_duplicates = len(all_perceptual_duplicates)
        total_robust_duplicates = len(all_robust_duplicates)
        
        # 2. Duplicate counts per dataset pair
        perceptual_pair_counts = {}
        robust_pair_counts = {}
        
        # Group duplicates by dataset pairs
        for dup in all_perceptual_duplicates:
            pair_key = f"{dup[0]['dataset']} - {dup[1]['dataset']}"
            if pair_key not in perceptual_pair_counts:
                perceptual_pair_counts[pair_key] = []
            perceptual_pair_counts[pair_key].append((dup[0]['name'], dup[1]['name']))
        
        for dup in all_robust_duplicates:
            pair_key = f"{dup[0]['dataset']} - {dup[1]['dataset']}"
            if pair_key not in robust_pair_counts:
                robust_pair_counts[pair_key] = []
            robust_pair_counts[pair_key].append((dup[0]['name'], dup[1]['name']))
        
        # Create detailed results sections for README
        perceptual_results = f"### Perceptual Hash Results\n\n"
        perceptual_results += f"**Total duplicate images found: {total_perceptual_duplicates}**\n\n"
        perceptual_results += "#### Duplicates by Dataset Pair\n\n"
        
        for pair, duplicates in perceptual_pair_counts.items():
            perceptual_results += f"- **{pair}**: {len(duplicates)} duplicates\n"
            # List all duplicates
            if duplicates:
                perceptual_results += "  - Full list of pairs:\n"
                for i, (img1, img2) in enumerate(duplicates):
                    perceptual_results += f"    - {img1} ↔ {img2}\n"
        
        robust_results = f"\n### Robust Hash Results\n\n"
        robust_results += f"**Total duplicate images found: {total_robust_duplicates}**\n\n"
        robust_results += "#### Duplicates by Dataset Pair\n\n"
        
        for pair, duplicates in robust_pair_counts.items():
            robust_results += f"- **{pair}**: {len(duplicates)} duplicates\n"
            # List all duplicates
            if duplicates:
                robust_results += "  - Full list of pairs:\n"
                for i, (img1, img2) in enumerate(duplicates):
                    robust_results += f"    - {img1} ↔ {img2}\n"
        
        # 3. Create examples folder and copy images
        examples_dir = os.path.join(main_analysis_dir, 'examples')
        os.makedirs(examples_dir, exist_ok=True)
        
        # Track all files to be copied to handle potential filename conflicts
        files_to_copy = {}
        
        # Process all duplicates for the table
        perceptual_pairs = []
        for dup in all_perceptual_duplicates:
            img1 = dup[0]
            img2 = dup[1]
            
            # Add to files to copy
            img1_filename = img1['name'] + ".jpg"
            img2_filename = img2['name'] + ".jpg"
            
            # Check for filename conflicts
            if img1_filename in files_to_copy and files_to_copy[img1_filename] != img1['path']:
                img1_filename = f"{img1['dataset']}_{img1_filename}"
            
            if img2_filename in files_to_copy and files_to_copy[img2_filename] != img2['path']:
                img2_filename = f"{img2['dataset']}_{img2_filename}"
            
            files_to_copy[img1_filename] = img1['path']
            files_to_copy[img2_filename] = img2['path']
            
            # Add to pairs list
            perceptual_pairs.append({
                'img1': {
                    'dataset': img1['dataset'],
                    'name': img1['name'],
                    'filename': img1_filename
                },
                'img2': {
                    'dataset': img2['dataset'],
                    'name': img2['name'],
                    'filename': img2_filename
                }
            })
        
        # Process robust duplicates
        robust_pairs = []
        for dup in all_robust_duplicates:
            img1 = dup[0]
            img2 = dup[1]
            
            # Add to files to copy
            img1_filename = img1['name'] + ".jpg"
            img2_filename = img2['name'] + ".jpg"
            
            # Check for filename conflicts
            if img1_filename in files_to_copy and files_to_copy[img1_filename] != img1['path']:
                img1_filename = f"{img1['dataset']}_{img1_filename}"
            
            if img2_filename in files_to_copy and files_to_copy[img2_filename] != img2['path']:
                img2_filename = f"{img2['dataset']}_{img2_filename}"
            
            files_to_copy[img1_filename] = img1['path']
            files_to_copy[img2_filename] = img2['path']
            
            # Add to pairs list
            robust_pairs.append({
                'img1': {
                    'dataset': img1['dataset'],
                    'name': img1['name'],
                    'filename': img1_filename
                },
                'img2': {
                    'dataset': img2['dataset'],
                    'name': img2['name'],
                    'filename': img2_filename
                }
            })
        
        # Copy all files to examples directory
        for filename, src_path in files_to_copy.items():
            dest_path = os.path.join(examples_dir, filename)
            try:
                import shutil
                shutil.copy2(src_path, dest_path)
                logger.info(f"Copied {src_path} to {dest_path}")
            except Exception as e:
                logger.error(f"Error copying {src_path} to {dest_path}: {e}")
        
        # Create a table of all image pairs
        all_pairs_table = """
## All Similar Image Pairs

Below are all pairs of similar images found across different datasets.

### Perceptual Hash Pairs

| Image 1 | Image 2 |
|---------|---------|
"""
        
        # Add all perceptual pairs to the table
        for pair in perceptual_pairs:
            img1 = pair['img1']
            img2 = pair['img2']
            all_pairs_table += f"""| ![{img1['name']}](examples/{img1['filename']}) <br> **Dataset:** {img1['dataset']} <br> **Filename:** {img1['name']} | ![{img2['name']}](examples/{img2['filename']}) <br> **Dataset:** {img2['dataset']} <br> **Filename:** {img2['name']} |\n"""
        
        # Add robust hash pairs
        all_pairs_table += """
### Robust Hash Pairs

| Image 1 | Image 2 |
|---------|---------|
"""
        
        # Add all robust pairs to the table
        for pair in robust_pairs:
            img1 = pair['img1']
            img2 = pair['img2']
            all_pairs_table += f"""| ![{img1['name']}](examples/{img1['filename']}) <br> **Dataset:** {img1['dataset']} <br> **Filename:** {img1['name']} | ![{img2['name']}](examples/{img2['filename']}) <br> **Dataset:** {img2['dataset']} <br> **Filename:** {img2['name']} |\n"""
        
        # Write README content
        readme_content = f"""# Cross-Dataset Image Similarity Analysis

## Overview

This analysis examines the similarity between images across different datasets using perceptual and robust hashing techniques. The focus is on identifying similar images between different datasets, rather than within the same dataset.

## Datasets Analyzed

The following datasets were analyzed:

{datasets_list}

## Methodology

Two different hashing techniques were used to compute image similarity:

1. **Perceptual Hashing ({perceptual_hash_type}, size={perceptual_hash_size}x{perceptual_hash_size})**: 
   - Detects visually similar images based on their appearance
   - Threshold for similarity: {perceptual_threshold}

2. **Robust Hashing (block mean hash, size={robust_hash_size}x{robust_hash_size})**: 
   - More robust to minor image transformations
   - Threshold for similarity: {robust_threshold}

## Results

### Dataset Overlap

The Venn diagrams show the overlap between the datasets:

- [Perceptual Hash Similarity Venn Diagram](./perceptual_venn_diagram.png)
- [Robust Hash Similarity Venn Diagram](./robust_venn_diagram.png)

The size of each circle is proportional to the number of images in the dataset, and the overlapping regions show the number of similar images between datasets.

### Cross-Dataset Duplicates

The following files contain information about cross-dataset similarities:

- **Consolidated duplicates**:
  - Perceptual hash duplicates: `perceptual_cross_duplicates.json`
  - Robust hash duplicates: `robust_cross_duplicates.json`

{all_pairs_table}

## Dataset-specific Files

Each dataset has its own directory with the following files:

{dataset_files_list}

## Summary

This analysis focused on cross-dataset image similarity, computing hashes for all images and comparing them across different datasets. The results provide insights into the overlap between datasets and can be used to identify duplicate or similar images across collections.
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"README.md created at {readme_path}")
        
        return {
            "datasets_processed": len(all_datasets),
            "dataset_names": [dataset['name'] for dataset in all_datasets],
            "files_generated": {
                "perceptual_cross_duplicates": perceptual_duplicates_path,
                "robust_cross_duplicates": robust_duplicates_path,
                "perceptual_venn_diagram": perceptual_venn_path if len(all_datasets) == 3 else None,
                "robust_venn_diagram": robust_venn_path if len(all_datasets) == 3 else None,
                "readme": readme_path
            }
        }
    else:
        # Only one dataset, no cross-dataset analysis
        logger.info("Only one dataset provided, skipping cross-dataset analysis")
        
        # Create summary
        summary = {
            "datasets_processed": len(all_datasets),
            "dataset_names": [dataset['name'] for dataset in all_datasets],
            "message": "Only one dataset was provided, so no cross-dataset analysis was performed."
        }
        
        # Save summary
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to dataset-specific analysis directories")
        
        return summary


if __name__ == '__main__':
    main()
