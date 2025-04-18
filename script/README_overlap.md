# Image Overlap Analysis and Benchmark Preparation Script

This script provides comprehensive functionality for analyzing image dataset overlap, detecting duplicates, and preparing benchmarks for few-shot learning. It uses CLIP embeddings and perceptual hashing to identify similar images across datasets and generates various visualizations to help understand the relationships between datasets.

## Features

- **CLIP Embedding Extraction**:
  - Extract visual features from images using various CLIP models (ViT-B/32, ViT-B/16, ViT-L/14)
  - Process images in batches for efficient computation

- **Duplicate Detection Methods**:
  - **Cosine Similarity**: Compare CLIP embeddings to find similar images
  - **Perceptual Hashing**: Use image hashing algorithms (phash, dhash, whash, ahash) to find visually similar images
  - Detect duplicates both within and across datasets

- **Embedding Projections and Visualizations**:
  - **t-SNE**: Project high-dimensional embeddings to 2D for visualization
  - **UMAP**: Alternative projection method that better preserves global structure
  - Interactive visualizations with Plotly for both individual and combined datasets
  - Venn diagrams showing dataset overlaps at different thresholds
  - Duplicate example visualizations showing pairs of similar images

- **Few-shot Sampling Strategies**:
  - **Random Sampling**: Basic k-shot sampling with random class selection
  - **Stratified Sampling**: Balanced sampling focusing on most frequent classes
  - **Long-tail Aware Sampling**: Focuses on underrepresented classes
  - Generate consistent splits with different random seeds for reproducibility

- **Cross-dataset Generalization Evaluation**:
  - Evaluate how well embeddings from one dataset transfer to another
  - Analyze class-level alignment between datasets
  - Visualize generalization performance

## Directory Structure

The script now organizes outputs in a more logical directory structure:

- **Dataset-specific analysis**: `dataset/{dataset}-data/analysis/`
  - Contains embeddings, projections, and within-dataset duplicate detection results
  - Each dataset has its own analysis directory

- **Cross-dataset analysis**: `dataset/cross/`
  - Contains cross-dataset duplicate detection results
  - Combined projections and visualizations
  - Venn diagrams showing dataset overlaps
  - Cross-dataset generalization evaluation results

## Usage

The script uses Click for command-line arguments:

```bash
python compute_overlap.py [OPTIONS]
```

### Options

- `--datasets`: List of datasets to analyze (default: ['ArtDL', 'ICONCLASS', 'wikidata'])
- `--max-images`: Maximum number of images to process per dataset
- `--clip-model`: CLIP model to use (default: 'ViT-B/32')
- `--batch-size`: Batch size for processing images (default: 32)
- `--method`: Method for duplicate detection ('cosine', 'perceptual', or 'both', default: 'both')
- `--cosine-threshold`: Similarity threshold for cosine-based duplicate detection (default: 0.9)
- `--hash-threshold`: Difference threshold for perceptual hash-based duplicate detection (default: 5)
- `--hash-size`: Size of perceptual hash (default: 8)
- `--hash-type`: Type of perceptual hash ('phash', 'dhash', 'whash', 'ahash', default: 'phash')
- `--projections`: Projection methods to use ('tsne', 'umap', or 'both', default: 'both')
- `--tsne-perplexity`: Perplexity parameter for t-SNE (default: 30)
- `--umap-neighbors`: Number of neighbors for UMAP (default: 15)
- `--use-ground-truth`: Use ground truth labels from 2_ground_truth.json for clustering
- `--generate-few-shot`: Generate few-shot splits
- `--k-shots`: Number of examples per class for few-shot sampling (default: [1, 5, 10])
- `--n-classes`: Number of classes for few-shot sampling (default: 10)
- `--n-seeds`: Number of random seeds for few-shot sampling (default: 3)
- `--evaluate-cross-dataset`: Evaluate cross-dataset generalization
- `--verbose`: Enable verbose logging (DEBUG level)
- `--help`: Show help message and exit

### Examples

Analyze all datasets with default settings:

```bash
python compute_overlap.py
```

Analyze specific datasets with cosine similarity only:

```bash
python compute_overlap.py --datasets ArtDL --datasets ICONCLASS --method cosine
```

Use perceptual hashing with custom threshold:

```bash
python compute_overlap.py --method perceptual --hash-threshold 3 --hash-type dhash
```

Use lower thresholds for both methods to find more potential duplicates:

```bash
python compute_overlap.py --method both --cosine-threshold 0.8 --hash-threshold 8
```

Generate few-shot splits:

```bash
python compute_overlap.py --generate-few-shot --k-shots 1 --k-shots 5 --n-classes 5
```

Use ground truth labels for clustering in visualizations:

```bash
python compute_overlap.py --use-ground-truth
```

Evaluate cross-dataset generalization:

```bash
python compute_overlap.py --evaluate-cross-dataset
```

## Output Files

### Dataset-specific Analysis (`dataset/{dataset}-data/analysis/`)

1. **Embedding files**:
   - `embeddings.npy`: NumPy array of CLIP embeddings
   - `names.json`: JSON file with image names
   - `labels.json`: JSON file with image labels
   - `hashes.json`: JSON file with perceptual hashes (if method includes 'perceptual')

2. **Projection visualizations**:
   - `tsne.html` and `tsne.png`: t-SNE projection of embeddings
   - `umap.html` and `umap.png`: UMAP projection of embeddings

3. **Duplicate detection**:
   - `cosine_duplicates.json`: List of duplicate pairs found using cosine similarity
   - `perceptual_duplicates.json`: List of duplicate pairs found using perceptual hashing
   - `{dataset}_duplicate_examples_cosine.png`: Visualization of cosine-based duplicates
   - `{dataset}_duplicate_examples_perceptual.png`: Visualization of perceptual hash-based duplicates

4. **Few-shot splits** (if requested):
   - `{dataset}_few_shot_splits/`: Directory containing few-shot splits
   - Various JSON and NumPy files for different sampling strategies, k values, and seeds

### Cross-dataset Analysis (`dataset/cross/`)

1. **Cross-dataset duplicates**:
   - `{dataset1}_{dataset2}_cosine_duplicates.json`: List of duplicate pairs between datasets using cosine similarity
   - `{dataset1}_{dataset2}_perceptual_duplicates.json`: List of duplicate pairs between datasets using perceptual hashing
   - `{dataset1}_{dataset2}_duplicate_examples_cosine.png`: Visualization of cross-dataset cosine-based duplicates
   - `{dataset1}_{dataset2}_duplicate_examples_perceptual.png`: Visualization of cross-dataset perceptual hash-based duplicates

2. **Combined projections**:
   - `combined_tsne_by_dataset.html` and `combined_tsne_by_dataset.png`: Combined t-SNE projection colored by dataset
   - `combined_tsne_by_label.html` and `combined_tsne_by_label.png`: Combined t-SNE projection colored by label
   - `combined_umap_by_dataset.html` and `combined_umap_by_dataset.png`: Combined UMAP projection colored by dataset
   - `combined_umap_by_label.html` and `combined_umap_by_label.png`: Combined UMAP projection colored by label

3. **Venn diagrams**:
   - `venn_diagram_cosine.png`: Venn diagram showing dataset overlaps using cosine similarity
   - `venn_diagram_perceptual.png`: Venn diagram showing dataset overlaps using perceptual hashing

4. **Cross-dataset generalization** (if requested):
   - `cross_dataset_{source}_{target}.json`: Generalization results from source to target dataset
   - `cross_dataset_{source}_{target}.png`: Visualization of generalization results

## Understanding the Results

### Duplicate Detection Methods

The script provides two methods for detecting duplicates:

1. **Cosine Similarity**:
   - Based on CLIP embeddings, which capture semantic content
   - Higher threshold (e.g., 0.9) means more similar images
   - Good for finding semantically similar images, even if they look different

2. **Perceptual Hashing**:
   - Based on image appearance, not semantic content
   - Lower threshold (e.g., 5) means more similar images
   - Good for finding visually similar images, even if they have different content

### Venn Diagrams and Dataset Overlap

The Venn diagrams provide a visual representation of the overlap between datasets:

1. **Separate Diagrams for Each Method**: The script generates different Venn diagrams for each duplicate detection method:
   - `venn_diagram_cosine.png`: Shows overlap based on cosine similarity of CLIP embeddings
   - `venn_diagram_perceptual.png`: Shows overlap based on perceptual hash differences

2. **Circle Size**: The size of each circle is proportional to the number of images in that dataset.

3. **Overlap Regions**: The overlapping regions represent images that are similar across datasets:
   - For two datasets A and B, the overlap shows images from A that have similar images in B (and vice versa).
   - For three datasets, the diagram shows all possible overlap combinations (A∩B, A∩C, B∩C, and A∩B∩C).

4. **Overlap Calculation**:
   - Two datasets: The overlap is the number of images from the first dataset that have similar images in the second dataset.
   - Three datasets: The triple overlap (A∩B∩C) represents images that are similar across all three datasets, calculated by finding images that appear in all three pairwise overlap sets.

5. **Proportional Representation**: The diagram ensures that:
   - No overlap region exceeds the size of the smallest dataset involved.
   - The total area of each circle accurately represents the dataset size.
   - The overlap regions are sized proportionally to the number of similar images.

6. **Interpretation**:
   - Large overlap regions indicate significant similarity between datasets.
   - Small or no overlap suggests the datasets contain distinct images.
   - Comparing the cosine and perceptual diagrams can reveal different types of similarity:
     - Cosine similarity captures semantic similarity (similar content)
     - Perceptual hashing captures visual similarity (similar appearance)
   - Datasets with high overlap in both diagrams likely contain near-identical images

### Embedding Projections

The embedding projections (t-SNE and UMAP) provide a way to visualize the relationships between images in a 2D space:

- **t-SNE**: Focuses on preserving local structure, good for seeing clusters
- **UMAP**: Better at preserving both local and global structure
- **Label Sources**: 
  - Default labels are derived from image paths or `labels.json` files
  - Ground truth labels can be used instead by enabling the `--use-ground-truth` option, which loads labels from `2_ground_truth.json` files
  - The projection titles indicate which label source is being used

In the projections, points that are close together represent similar images. The projections can be colored by dataset or by label to see how images from different datasets or with different labels relate to each other. Using ground truth labels often provides more accurate and meaningful clusters, especially for datasets where the default labeling method might not capture the true semantic categories.

### Few-shot Sampling Strategies

The few-shot sampling strategies provide different ways to select examples for few-shot learning:

- **Random Sampling**: Simple random selection of classes and examples
- **Stratified Sampling**: Focuses on the most frequent classes
- **Long-tail Aware Sampling**: Gives more weight to rare classes

These strategies can be used to create consistent few-shot splits for benchmarking few-shot learning algorithms.

### Cross-dataset Generalization

The cross-dataset generalization evaluation measures how well embeddings from one dataset transfer to another. This is useful for understanding how models trained on one dataset might perform on another.

## Conclusions from Cross-dataset Analysis

Analysis of the cross-dataset results reveals several important insights:

1. **Dataset Overlap**: The Venn diagrams in the cross directory show significant overlap between the ArtDL, ICONCLASS, and wikidata datasets, particularly at lower similarity thresholds. This suggests that these datasets contain many similar images, which could lead to data leakage if used naively for training and evaluation.

2. **Duplicate Detection Methods**: Comparing the cosine similarity and perceptual hashing results shows that:
   - Cosine similarity tends to find more semantically similar images (e.g., same subject but different style)
   - Perceptual hashing finds more visually similar images (e.g., same image with minor modifications)
   - Using both methods provides a more comprehensive view of dataset overlap

3. **Embedding Projections**: The combined projections show that:
   - Images from the same dataset tend to cluster together, indicating dataset-specific biases
   - When colored by label, we can see that some classes form clear clusters across datasets, while others are more scattered
   - This information can be used to identify which classes might benefit from cross-dataset training

4. **Cross-dataset Generalization**: The generalization results indicate that:
   - Some classes transfer well between datasets (high centroid similarity)
   - Others show poor transfer (low centroid similarity)
   - This information can guide the selection of source datasets for transfer learning

5. **Implications for Benchmarking**:
   - The few-shot splits should be created with awareness of dataset overlap
   - Cross-dataset evaluation should account for the varying degrees of generalization across classes
   - The long-tail aware sampling strategy may be particularly useful for creating challenging few-shot benchmarks

These insights highlight the importance of careful dataset analysis before using them for training and evaluation. The tools provided by this script enable researchers to make informed decisions about dataset selection, preprocessing, and benchmark creation.
