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
- `system_prompt_{dataset}_[description]_[enhanced].txt`

For example:
- `system_prompt_artdl.txt`: Basic prompt for ArtDL dataset
- `system_prompt_artdl_description.txt`: Prompt using class descriptions for ArtDL
- `system_prompt_artdl_enhanced.txt`: Enhanced version of the basic prompt

#### Caching System

The script implements a caching system to avoid redundant API calls:
- Cache files are stored in `gpt_data/cache_{model}_{dataset}_{test}.json`
- Each image ID is mapped to its classification probabilities
- The cache is loaded at the start and saved periodically during processing

### execute_clip.py

This script uses OpenAI's CLIP models to classify images into predefined categories.

#### Usage

```bash
python execute_clip.py --models clip-vit-base-patch32 clip-vit-base-patch16 --datasets ArtDL IconArt --folders test_1 test_2
```

### execute_siglip.py

This script uses Google's SigLIP models to classify images into predefined categories.

### evaluate.py

This script evaluates the results of the classification models using standard metrics.

#### Usage

```bash
python evaluate.py --models clip-vit-base-patch32 gpt-4o --folders test_1 test_2 --datasets ArtDL
```

## Test Configurations

The scripts support different test configurations:

- `test_1`: Uses class labels for classification
- `test_2`: Uses class descriptions for classification
- `test_3`: Uses few-shot learning with class labels
- `test_4`: Uses few-shot learning with class descriptions

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
