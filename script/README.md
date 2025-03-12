# LLM-test Script Documentation

This directory contains scripts for evaluating different image classification models on art datasets.

## Overview

The scripts in this directory are designed to:

1. Execute different models (GPT, CLIP, SigLIP) on image classification tasks
2. Evaluate the results using standard metrics
3. Support different test configurations and datasets

## Scripts

### execute_gpt.py

This script uses OpenAI's GPT models with vision capabilities to classify images into predefined categories.

#### Features

- Supports multiple GPT models (gpt-4o, gpt-4o-mini)
- Handles different datasets and test configurations
- Implements caching to save API calls and costs
- Provides cost estimation for API usage
- Supports few-shot learning with example images

#### Usage

```bash
python execute_gpt.py --models gpt-4o gpt-4o-mini --datasets ArtDL IconArt --folders test_1 test_2
```

#### Parameters

- `--models`: List of GPT model names to use (e.g., gpt-4o, gpt-4o-mini)
- `--folders`: List of test folders to use (default: test_1, test_2)
- `--datasets`: List of datasets to use (default: ArtDL)
- `--limit`: Limit the number of images to process (-1 for all)
- `--batch_size`: Number of images per batch (default: 10)
- `--save_frequency`: How often to save cache in batches (default: 5)

#### Configuration

The script requires an OpenAI API key, which should be stored in `gpt_data/psw.ini` in the following format:

```ini
[openai]
api_key=your_api_key_here
```

#### System Prompts

System prompts are stored in the `gpt_data` directory with the following naming convention:
- `prompts/{dataset}/{test}.txt`

#### Caching System

The script implements a caching system to avoid redundant API calls:
- Cache files are stored in `gpt_data/cache/{dataset}/{test}/{model}.json`
- Each image ID is mapped to its classification probabilities
- The cache is loaded at the start and saved periodically during processing

### execute_clip.py

This script uses OpenAI's CLIP models to classify images into predefined categories.

#### Features

- Supports multiple CLIP models (clip-vit-base-patch32, clip-vit-base-patch16, clip-vit-large-patch14)
- Handles different datasets and test configurations
- Processes images in batches for efficient classification
- Supports both zero-shot classification (test_1, test_2) and fine-tuned models (test_3, test_4)

#### Usage

```bash
python execute_clip.py --models clip-vit-base-patch32 clip-vit-base-patch16 --datasets ArtDL IconArt --folders test_1 test_2
```

#### Parameters

- `--models`: List of CLIP model names to use (e.g., clip-vit-base-patch32, clip-vit-base-patch16)
- `--folders`: List of test folders to use (default: test_1, test_2, test_3)
- `--datasets`: List of datasets to use (default: ArtDL, IconArt)
- `--verbose`: Enable verbose logging (DEBUG level)

### execute_siglip.py

This script uses Google's SigLIP models to classify images into predefined categories.

#### Features

- Supports multiple SigLIP models (siglip-base-patch16-512, siglip-large-patch16-384, siglip-so400m-patch14-384)
- Handles different datasets and test configurations
- Processes images in batches for efficient classification
- Supports both zero-shot classification and fine-tuned models

#### Usage

```bash
python execute_siglip.py --models siglip-base-patch16-512 siglip-large-patch16-384 --datasets ArtDL IconArt --folders test_1 test_2
```

#### Parameters

- `--models`: List of SigLIP model names to use (default: siglip-base-patch16-512, siglip-large-patch16-384, siglip-so400m-patch14-384)
- `--folders`: List of test folders to use (default: test_1, test_2, test_3)
- `--datasets`: List of datasets to use (default: ArtDL, IconArt)

### few-shot.py

This script fine-tunes CLIP and SigLIP models using a few-shot learning approach for image classification.

#### Features

- Supports fine-tuning of both CLIP and SigLIP models
- Uses a small set of example images for training
- Freezes most model layers and only fine-tunes the last transformer layers
- Saves the fine-tuned models for later use in test_3 and test_4 configurations

#### Usage

```bash
python few-shot.py --models clip-vit-base-patch32 siglip-base-patch16-512 --datasets ArtDL --folders test_3 --num_epochs 150 --lr 1e-5
```

#### Parameters

- `--models`: List of model names to train (default: clip-vit-base-patch32, siglip-base-patch16-512)
- `--folders`: List of input folders (default: test_3)
- `--num_epochs`: Number of epochs to train (default: 150)
- `--lr`: Learning rate (default: 1e-5)
- `--datasets`: List of datasets to use (default: ArtDL)

### evaluate.py

This script evaluates the results of the classification models using standard metrics.

#### Features

- Calculates confusion matrices, precision, recall, F1 scores, and average precision
- Generates visualizations of confusion matrices
- Computes both per-class and overall (macro/micro) metrics
- Saves results in multiple formats (CSV, Markdown, PNG)

#### Usage

```bash
python evaluate.py --models clip-vit-base-patch32 gpt-4o --folders test_1 test_2 --datasets ArtDL
```

#### Parameters

- `--models`: List of models to evaluate (default: all CLIP and SigLIP models)
- `--folders`: List of folders to evaluate (default: test_1, test_2, test_3)
- `--limit`: Limit the number of images to evaluate (-1 for all)
- `--datasets`: List of datasets to use (default: ArtDL, IconArt)

### logger_utils.py

This utility module provides consistent logging functionality for the classification scripts.

#### Features

- Sets up loggers with consistent formatting
- Handles logging without disrupting tqdm progress bars
- Supports both file and console logging
- Configurable verbosity levels

#### Usage

```python
import logger_utils

# Setup logger for a specific dataset/test/model combination
logger = logger_utils.setup_logger(dataset, test, model, output_folder, verbose=False)

# Use the logger
logger.info("Processing started")
logger.debug("Detailed information")
logger.error("An error occurred")
```

## Test Configurations

The scripts support different test configurations:

- `test_1`: Uses class labels for classification
- `test_2`: Uses class descriptions for classification
- `test_3`: Uses few-shot learning with class labels
- `test_4`: Uses few-shot learning with class descriptions **DEPRECATED**

## Code Structure

### execute_gpt.py

The script is organized into the following components:

1. **ModelConfig**: Configuration for different GPT models including pricing
2. **CacheManager**: Manages caching of API responses to avoid redundant calls
3. **GPTImageClassifier**: Main class for classifying images using GPT models
4. **Helper Functions**:
   - `encode_image`: Encodes an image as a base64 data URL
   - `load_images`: Loads images from disk
5. **Main Function**: Orchestrates the classification process

### Class Hierarchy

```
ModelConfig
  ├── MODELS (dict): Configuration for different models
  └── get_costs(): Get costs for a specific model

CacheManager
  ├── _load_cache(): Load cache from disk
  ├── get_result(): Get cached result for an image
  ├── add_result(): Add a new result to the cache
  └── save(): Save cache to disk

GPTImageClassifier
  ├── _get_prompt_path(): Get the path to the appropriate prompt file
  ├── _load_few_shot_examples(): Load few-shot examples
  ├── _prepare_batch_request(): Prepare the API request for a batch
  ├── _parse_response(): Parse the API response
  ├── classify_images(): Main classification method
  └── _display_cost_info(): Calculate and display cost information
```

## Output

The scripts save their output in the following structure:

```
{test_folder}/{dataset}/{model}/
  ├── probs.npy: NumPy array of class probabilities
  ├── confusion_matrix.csv: Confusion matrix as CSV
  ├── confusion_matrix.png: Visualization of confusion matrix
  ├── class_metrics.csv: Per-class metrics
  ├── class_metrics.md: Per-class metrics in Markdown format
  └── summary_metrics.csv: Overall metrics
```
