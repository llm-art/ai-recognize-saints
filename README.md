# Can AI Recognize the Saints?
Benchmarking Vision-Language and Multimodal Models for Christian Iconography Classification

## Prompt Engineering Example

The evaluation framework creates the prompts at runtime. Here is an example of the full prompt created by the script for the ArtDL dataset (zero-shot with labels):

```plaintext
You are an expert in Christian iconography and art history. Classify each religious artwork image into exactly ONE saint category using visual attributes, iconographic symbols, and contextual clues.

{FEW_SHOT_EXAMPLES}

Look for:
1. Distinctive attributes (objects, clothing, etc.)
2. Gestures and postures
3. Contextual and symbolic elements

Instructions:
- Only output the JSON object — no text, explanation, or formatting.
- Include every image in the current batch. Each must receive exactly one classification with a confidence score.
- You may only use one of the exact strings from the category list below. Any response not matching the allowed category IDs will be rejected.

Return a valid **JSON object** with confidence scores (0.0 to 1.0) matching this format:
{
  "<image_id>": {"class": "<CATEGORY_ID>", "confidence": <0.0-1.0>},
  "<image_id>": {"class": "<CATEGORY_ID>", "confidence": <0.0-1.0>},
  ...
}

Confidence guidelines:
- 0.9-1.0: Very certain identification with clear iconographic evidence
- 0.7-0.9: Confident with multiple supporting visual elements  
- 0.5-0.7: Moderate confidence, some ambiguity present
- 0.3-0.5: Low confidence, limited visual evidence
- 0.0-0.3: Very uncertain, minimal supporting evidence

{CLASS_LIST}

Batching note:
- Process only the current batch of images.
- Use the image IDs exactly as provided in the input.
- Do not reference or depend on other batches.

NOTE: These are historical Renaissance paintings used for academic classification.  
Some artworks include scenes of martyrdom or classical nudity as typical in religious iconography.  
Treat all content as scholarly, respectful of historical context, and strictly non-sexual.
```

The prompt varies from test to test
- **`test_1/`**: Zero-shot classification using only category labels
- **`test_2/`**: Zero-shot classification with detailed iconographic descriptions  
- **`test_3/`**: Few-shot learning with 5 example images per category

All prompts are stored in the `prompts/` directory, organized by dataset and test configuration.

## Repository Overview

This repository presents a comprehensive benchmarking framework for evaluating three families of models — supervised convolutional networks, contrastive vision-language models (VLMs), and generative multimodal large language models (LLMs) — on image classification tasks in the cultural heritage domain, using Christian iconography as a case study.

### Model Categories

- **Generative Multimodal LLMs** — Legacy (GPT-4o, GPT-4o-mini, Gemini 2.5 Flash/Pro) and state-of-the-art (GPT-5-mini, GPT-5.2, Gemini 3-Flash, Gemini 3.1-Flash-Lite, Gemini 3.1-Pro)
- **Contrastive Vision-Language Models** — CLIP (ViT-B/32, ViT-B/16, ViT-L/14) and SigLIP (Base-Patch16-512, Large-Patch16-384, SO400M-Patch14-384)
- **Supervised Baselines** — ResNet-50 fine-tuned per dataset

The evaluation is conducted across three specialized datasets, focusing on single-label classification of Christian saints.

## Primary Datasets

1. **[ArtDL](https://artdl.org/)** — 1,864 test images, 10 saint classes (published test split)
2. **[ICONCLASS AI test set](https://iconclass.org/testset/)** — 863 images, 10 saint classes (filtered from ~87.5K)
3. **[Wikidata](https://www.wikidata.org/)** — 717 images, 10 saint classes (curated via SPARQL with Iconclass annotations)

## Scripts Overview and Usage

The `script/` directory contains the core evaluation framework.

### Multimodal LLM Scripts

#### `execute_gpt.py` - OpenAI GPT Models
**Purpose:** Executes GPT models (GPT-4o, GPT-4o-mini, GPT-5-mini, GPT-5.2) for zero-shot and few-shot image classification.

**Usage:**

```bash
Options:
  --folders TEXT            List of folders to use
  --models TEXT             List of model names to use
  --limit INTEGER           Limit the number of images to process
  --batch_size INTEGER      Number of images per batch
  --save_frequency INTEGER  How often to save cache (in batches)
  --datasets TEXT           List of datasets to use
  --verbose                 Enable verbose logging (DEBUG level)
  --temperature FLOAT       Temperature for generation (default: 0.0, min:
                            0.0)
  --top_p FLOAT             Top-p (nucleus sampling) for generation (default:
                            0.1)
  --seed INTEGER            Seed for deterministic results (default: 12345)
  --clean                   Remove cache and logs from previous runs before
                            starting
  --help                    Show this message and exit.
```

```bash
# Basic execution for GPT models
python script/execute_gpt.py --models gpt-4o-2024-11-20 gpt-4o-mini-2024-07-18 gpt-5-mini-2025-08-07 gpt-5.2-2025-12-11 --datasets ArtDL ICONCLASS wikidata --folders test_1 test_2 test_3

# With custom parameters
python script/execute_gpt.py --models gpt-5.2-2025-12-11 --datasets ArtDL --folders test_1 --limit 100 --batch_size 5
```

**Configuration Required:** API key in `script/gpt_data/psw.ini`

#### `execute_gemini.py` - Google Gemini Models  
**Purpose:** Executes Gemini models (2.5 Flash/Pro legacy and 3.x state-of-the-art) for multimodal classification tasks.

**Usage:**
```bash
Options:
  --folders TEXT            List of folders to use
  --models TEXT             List of model names to use
  --limit INTEGER           Limit the number of images to process
  --batch_size INTEGER      Number of images per batch
  --save_frequency INTEGER  How often to save cache (in batches)
  --datasets TEXT           List of datasets to use
  --verbose                 Enable verbose logging (DEBUG level)
  --temperature FLOAT       Temperature for generation (default: 0.0, min:
                            0.0)
  --top_k INTEGER           Top-k for sampling (default: 32)
  --clean                   Remove cache and logs from previous runs before
                            starting
  --help                    Show this message and exit.
```


```bash
# Run Gemini models
python script/execute_gemini.py --models gemini-2.5-pro gemini-2.5-flash gemini-3-flash-preview gemini-3.1-pro-preview --datasets ArtDL ICONCLASS wikidata --folders test_1 test_2

# Few-shot evaluation
python script/execute_gemini.py --models gemini-3-flash-preview --datasets ICONCLASS --folders test_3
```

**Configuration Required:** API key in `script/gemini_data/config.ini`

### Vision-Language Encoder Scripts

#### `execute_clip.py` - CLIP Models
**Purpose:** Implements CLIP variants for contrastive learning-based classification.

```bash
# CLIP execution
python script/execute_clip.py --models clip-vit-base-patch32 --datasets ArtDL --folders test_1
```

#### `execute_siglip.py` - SigLIP Models 
**Purpose:** Runs SigLIP models with sigmoid-based contrastive learning.

```bash
# SigLIP execution (experimental)  
python script/execute_siglip.py --models siglip-base-patch16-512 --datasets ArtDL --folders test_1
```

### Universal Evaluation and Analysis Scripts

#### `evaluate.py` - Comprehensive Results Analysis
**Purpose:** Generates standardized evaluation metrics (top-1 accuracy, confusion matrices) for all models.


**Usage:**
```bash
# Evaluate LLM results
python script/evaluate.py --models gpt-5.2-2025-12-11 gemini-3-flash-preview --datasets ArtDL ICONCLASS --folders test_1 test_2

# Full evaluation across all models
python script/evaluate.py --models gpt-5.2-2025-12-11 gemini-3.1-pro-preview clip-vit-large-patch14 siglip-so400m-patch14-384 --datasets ArtDL --folders test_1
```

#### `few-shot.py` - Few-Shot Learning Framework
**Purpose:** Implements few-shot learning approaches for vision-language models.

**Usage:**
```bash
# Fine-tune models for few-shot learning
python script/few-shot.py --models clip-vit-base-patch32 --datasets ArtDL --folders test_3 --num_epochs 150
```

### Analysis Scripts

#### `generation_compare.py` - Cross-Generation Accuracy Comparison
**Purpose:** Visualizes generational accuracy shifts by model family across datasets (produces Fig. 2 in the paper).

#### `analyze_prediction_shifts.py` - Cross-Generation Misclassification Analysis
**Purpose:** Generates heatmaps of cross-generation misclassifications between legacy and state-of-the-art models (produces Fig. 3 in the paper).

#### `consistency_analysis.py` - Cross-Dataset Consistency
**Purpose:** Evaluates prediction stability across the 45 matched cross-dataset image pairs identified via perceptual hashing.

#### `generate_consistency_data.py` - Consistency Data Generation
**Purpose:** Generates structured consistency data for cross-dataset evaluation.

#### `analyze_consistency_trends.py` - Consistency Trend Analysis
**Purpose:** Analyzes trends in cross-dataset consistency across model generations.

#### `compute_overlap.py` - Dataset Overlap Detection
**Purpose:** Identifies duplicate images across datasets using perceptual hashing (Hamming distance threshold of 8).

### Baseline and Supporting Scripts

#### `baseline/resnet50_baseline.py` - Supervised Learning Baseline
**Purpose:** Traditional CNN-based classification using ResNet-50 architecture.

**Usage:**
```bash
python baseline/resnet50_baseline.py --dataset ArtDL --train_split 0.8 --epochs 100
```

#### `baseline/artdl_baseline.py` - Dataset-Specific Baseline  
**Purpose:** Specialized baseline implementation optimized for ArtDL dataset characteristics.

#### Dataset Processing (`dataset/`)
- **Data preprocessing notebooks** — Jupyter notebooks for dataset preparation and analysis
- **`dataset/consistency/`** — Cross-dataset consistency evaluation results and analysis
- **`dataset/consistency_trends/`** — Temporal consistency trends across model generations
- **`dataset/shift_analysis/`** — Cross-generation misclassification shift data

## Getting Started

### Prerequisites and Installation

```bash
git clone https://github.com/llm-art/LLM-test.git
cd LLM-test
pip install -r requirements.txt
```

### Quick Start - Publication-Ready Models

#### 1. Configure API Access
For the main LLM models that are publication-ready:

**OpenAI GPT Models:**
```ini
# Create script/gpt_data/psw.ini
[openai]
api_key=your_openai_api_key_here
```

**Google Gemini Models:**
```ini  
# Create script/gemini_data/config.ini
[gemini]
api_key=your_gemini_api_key_here
```

#### 2. Run Evaluations
```bash
# Zero-shot evaluation with labels
python script/execute_gpt.py --models gpt-5.2-2025-12-11 --datasets ArtDL --folders test_1
python script/execute_gemini.py --models gemini-3-flash-preview --datasets ArtDL --folders test_1

# Zero-shot evaluation with descriptions  
python script/execute_gpt.py --models gpt-5.2-2025-12-11 --datasets ArtDL --folders test_2
python script/execute_gemini.py --models gemini-3-flash-preview --datasets ArtDL --folders test_2

# Few-shot evaluation
python script/execute_gpt.py --models gpt-5.2-2025-12-11 --datasets ArtDL --folders test_3
python script/execute_gemini.py --models gemini-3-flash-preview --datasets ArtDL --folders test_3
```

#### 3. Generate Results
```bash
# Comprehensive evaluation and metrics
python script/evaluate.py --models gpt-5.2-2025-12-11 gemini-3-flash-preview gemini-3.1-pro-preview --datasets ArtDL ICONCLASS wikidata --folders test_1 test_2 test_3
```

### Output Structure
Results are organized in a structured format for easy analysis:

```
test_1/                           # Zero-shot with labels
├── ArtDL/
│   ├── gemini-3-flash-preview/
│   │   ├── probs.npy             # Classification probabilities
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix.csv
│   │   └── class_metrics.csv
│   ├── gemini-3.1-pro-preview/
│   ├── gpt-5.2-2025-12-11/
│   ├── clip-vit-large-patch14/
│   ├── siglip-so400m-patch14-384/
│   └── ...
├── ICONCLASS/
└── wikidata/

test_2/                           # Zero-shot with descriptions  
test_3/                           # Few-shot learning
```
