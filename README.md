# Benchmarking Multimodal Large Language Models in Zero-shot and Few-shot Scenarios: A study on Christian Iconography

This repository contains the code for the following paper:
**Benchmarking Multimodal Large Language Models in Zero-shot and Few-shot Scenarios: preliminary results on studying Christian Iconography**.

## Prompt Engineering Example

The evaluation framework creates the prompts at runtime. Here is an example of the full prompt created by the script for the ArtDL dataset (zero-shot with labels):

```plaintext
You are an expert in Christian iconography and art history. Classify each religious artwork image into exactly ONE saint category using visual attributes, iconographic symbols, and contextual clues.



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

Each <CATEGORY_ID> must be one of (use only the category ID as output):

"antony_of_padua" - Antony of Padua
"john_baptist" - John the Baptist
"paul" - Paul
"francis" - Francis of Assisi
"mary_magdalene" - Mary Magdalene
"jerome" - Jerome
"dominic" - Saint Dominic
"mary" - Virgin Mary
"peter" - Peter
"sebastian" - Saint Sebastian

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

This repository presents a comprehensive evaluation framework for zero-shot and few-shot image classification approaches on Christian iconography datasets. The primary focus is on **Multimodal Large Language Models (MLLMs)**

### Model Categories

- **Large Multimodal Models** (GPT-4o, Gemini 2.5) - Complete evaluation with validated results
- **Traditional Supervised Baselines** (ResNet-50) - Established baseline comparisons

**Uncompleted Status:**
- **Vision-Language Encoders** (CLIP, SigLIP, BLIP2) - Not all results available

The evaluation is conducted across three specialized datasets, focusing on Christian iconography and the classification of religious art.

## Primary Datasets

1. **[ArtDL](https://artdl.org/)** - ~42.5K images with 10 classes of Christian icons
2. **[ICONCLASS AI test set](https://iconclass.org/testset/)** - ~87.5K images with Iconclass classification IDs
3. **[Wikidata](https://www.wikidata.org/)** - Curated dataset via SPARQL queries with ICONCLASS annotations

## Scripts Overview and Usage

The `script/` directory contains the core evaluation framework.

### Multimodal LLM Scripts

#### `execute_gpt.py` - OpenAI GPT Models
**Purpose:** Executes GPT-4o and GPT-4o-mini models for zero-shot and few-shot image classification.

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
python script/execute_gpt.py --models gpt-4o gpt-4o-mini --datasets ArtDL ICONCLASS wikidata --folders test_1 test_2 test_3

# With custom parameters
python script/execute_gpt.py --models gpt-4o --datasets ArtDL --folders test_1 --limit 100 --batch_size 5
```

**Configuration Required:** API key in `script/gpt_data/psw.ini`

#### `execute_gemini.py` - Google Gemini Models  
**Purpose:** Executes Gemini 2.5 Pro and Flash models for multimodal classification tasks.

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
python script/execute_gemini.py --models gemini-2.5-pro gemini-2.5-flash --datasets ArtDL ICONCLASS wikidata --folders test_1 test_2

# Few-shot evaluation
python script/execute_gemini.py --models gemini-2.5-pro --datasets ICONCLASS --folders test_3
```

**Configuration Required:** API key in `script/gemini_data/config.ini`

Vision-Language Encoder Scripts

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
**Purpose:** Generates standardized evaluation metrics for all models.


**Usage:**
```bash
# Evaluate LLM results
python script/evaluate.py --models gpt-4o gemini-2.5-pro --datasets ArtDL ICONCLASS --folders test_1 test_2

# Full evaluation
python script/evaluate.py --models gpt-4o gemini-2.5-pro clip-vit-base-patch32 --datasets ArtDL --folders test_1
```

#### `few-shot.py` - Few-Shot Learning Framework
**Purpose:** Implements few-shot learning approaches for vision-language models.

**Usage:**
```bash
# Fine-tune models for few-shot learning
python script/few-shot.py --models clip-vit-base-patch32 --datasets ArtDL --folders test_3 --num_epochs 150
```

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
- **Data preprocessing notebooks** - Jupyter notebooks for dataset preparation and analysis
- **Cross-dataset analysis tools** - Scripts for analyzing overlaps between datasets
- **Consistency evaluation framework** - Tools for model consistency analysis

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

#### 2. Run Publication-Ready Evaluations
```bash
# Zero-shot evaluation with labels
python script/execute_gpt.py --models gpt-4o --datasets ArtDL --folders test_1
python script/execute_gemini.py --models gemini-2.5-pro --datasets ArtDL --folders test_1

# Zero-shot evaluation with descriptions  
python script/execute_gpt.py --models gpt-4o --datasets ArtDL --folders test_2
python script/execute_gemini.py --models gemini-2.5-pro --datasets ArtDL --folders test_2

# Few-shot evaluation
python script/execute_gpt.py --models gpt-4o --datasets ArtDL --folders test_3
python script/execute_gemini.py --models gemini-2.5-pro --datasets ArtDL --folders test_3
```

#### 3. Generate Publication Results
```bash
# Comprehensive evaluation and metrics
python script/evaluate.py --models gpt-4o gemini-2.5-pro --datasets ArtDL ICONCLASS wikidata --folders test_1 test_2 test_3
```

### Output Structure
Results are organized in a structured format for easy analysis:

```
test_1/                    # Zero-shot with labels
├── ArtDL/
│   ├── gpt-4o/
│   │   ├── probs.npy      # Classification probabilities
│   │   ├── confusion_matrix.csv
│   │   └── class_metrics.csv
│   └── gemini-2.5-pro/
└── ICONCLASS/

test_2/                    # Zero-shot with descriptions  
test_3/                    # Few-shot learning
```

## Models Evaluated

### LLM Models

The following models have been thoroughly evaluated and validated for publication:

| Model Name        | Type                     | Input Context Window     | Output Tokens     | Open Source | Release Date | Knowledge Cut-off |
|-------------------|--------------------------|--------------------------|-------------------|--------------|---------------|--------------------|
| GPT-4o            | Multimodal LLM            | 128k tokens              | 16.4k tokens     | No         | Aug 2024      | Oct 2023           |
| GPT-4o mini       | Multimodal LLM            | 128k tokens              | 16.4k tokens     | No         | Jul 2024      | Oct 2023           |
| Gemini 2.5 Pro    | Multimodal LLM            | 1M tokens                | 64k tokens    | No         | Mar 2024      | Jan 2025   |
| Gemini 2.5 Flash  | Multimodal LLM            | 1M tokens                | 65k tokens    | No         | Apr 2025      | Jan 2025    |


## Publication-Ready Results

### ArtDL Dataset Results
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-------------------|
| **gpt-4o-2024-08-06**          | **86.00%**           | **87.45%**                 | **86.48%**         |
| **gpt-4o-mini-2024-07-18**     | **82.46%**           | **84.98%**                 | **84.60%**         |
| **gemini-2.5-flash-preview**   | **88.20%**           | **87.02%**                 | **84.71%**         |
| **gemini-2.5-pro-preview**     | **90.45%**           | **90.18%**                 | **86.59%**         |
| Baseline (ResNet-50)           | 84.44%               | -                          | -                  |

### ICONCLASS Dataset Results
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-------------------|
| **gpt-4o-2024-08-06**          | **75.32%**           | **75.43%**                 | **73.46%**         |
| **gpt-4o-mini-2024-07-18**     | **55.74%**           | **59.56%**                 | **55.50%**         |
| **gemini-2.5-flash-preview**   | **77.17%**           | **77.75%**                 | **78.22%**         |
| **gemini-2.5-pro-preview**     | **83.31%**           | **84.82%**                 | **84.59%**         |
| Baseline (ResNet-50)           | 40.46%               | -                          | -                  |

### Wikidata Dataset Results 
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-------------------|
| **gpt-4o-2024-08-06**          | **45.75%**           | **45.31%**                 | **45.31%**         |
| **gpt-4o-mini-2024-07-18**     | **35.78%**           | **36.95%**                 | **34.31%**         |
| **gemini-2.5-flash-preview**   | **45.45%**           | **45.31%**                 | **44.57%**         |
| **gemini-2.5-pro-preview**     | **45.89%**           | **45.31%**                 | **47.07%**         |
| Baseline (ResNet-50)           | 43.97%               | -                          | -                  |


## Citation

todo
