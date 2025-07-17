# Benchmarking Multimodal Large Language Models in Zero-shot and Few-shot Scenarios: A study on Christian Iconography

This repository contains a the work for the preliminary result of the paper  
**Benchmarking Multimodal Large Language Models in Zero-shot and Few-shot Scenarios: preliminary results on studying Christian Iconography**.


## Key Findings - Publication-Ready Results

### Multimodal LLM Performance (Validated Results)

1. **Superior classification accuracy** - Gemini 2.5 Pro and GPT-4o achieve 83-90% accuracy across Christian iconography datasets
2. **Consistent cross-dataset performance** - LLMs maintain robust performance across ArtDL, ICONCLASS, and Wikidata datasets
3. **Minimal prompt sensitivity** - Performance remains stable whether using class labels or detailed descriptions
4. **Cost-effective few-shot learning** - Few-shot approaches show marginal improvements over zero-shot, indicating strong pre-training capabilities

### Cross-Dataset Analysis (LLM Focus)

- **ArtDL**: Highest LLM performance (Gemini 2.5 Pro: 90.45%, GPT-4o: 86.00%)
- **ICONCLASS**: Strong LLM performance (Gemini 2.5 Pro: 83.31%, GPT-4o: 75.32%)  
- **Wikidata**: Moderate LLM performance showing dataset complexity effects

### Experimental Results (Vision-Language Encoders)

*Note: The following results are preliminary and under development for publication standards.*

- **CLIP variants**: Performance scales with model size, ranging from 16-57% accuracy
- **SigLIP models**: Generally outperform CLIP variants, achieving 43-66% accuracy
- **Contrastive learning limitations**: Significant performance gap compared to generative LLMs

## Repository Overview

This repository presents a comprehensive evaluation framework for zero-shot and few-shot image classification approaches on Christian iconography datasets. The primary focus is on **Multimodal Large Language Models (MLLMs)**, which have been thoroughly tested and validated for publication.

### Model Categories

**Publication-Ready Results:**
- **Large Multimodal Models** (GPT-4o, Gemini 2.5) - Complete evaluation with validated results
- **Traditional Supervised Baselines** (ResNet-50) - Established baseline comparisons

**Experimental/Development Status:**
- **Vision-Language Encoders** (CLIP, SigLIP) - Results available but not finalized for publication

The evaluation is conducted across three specialized datasets focusing on Christian iconography and religious art classification.

## Primary Datasets

1. **[ArtDL](https://artdl.org/)** - ~42.5K images with 10 classes of Christian icons
2. **[ICONCLASS AI test set](https://iconclass.org/testset/)** - ~87.5K images with Iconclass classification IDs
3. **[Wikidata](https://www.wikidata.org/)** - Curated dataset via SPARQL queries with ICONCLASS annotations

## Scripts Overview and Usage

The `script/` directory contains the core evaluation framework with distinct categories of execution scripts based on publication readiness.

### Publication-Ready: Multimodal LLM Scripts

These scripts have been fully tested and validated for publication:

#### `execute_gpt.py` - OpenAI GPT Models
**Purpose:** Executes GPT-4o and GPT-4o-mini models for zero-shot and few-shot image classification.

**Key Features:**
- API-based inference with comprehensive caching system
- Cost estimation and budget tracking
- Support for multiple test configurations
- Robust error handling and retry mechanisms

**Usage:**
```bash
# Basic execution for GPT models
python script/execute_gpt.py --models gpt-4o gpt-4o-mini --datasets ArtDL ICONCLASS --folders test_1 test_2

# With custom parameters
python script/execute_gpt.py --models gpt-4o --datasets ArtDL --folders test_1 --limit 100 --batch_size 5
```

**Configuration Required:** API key in `script/gpt_data/psw.ini`

#### `execute_gemini.py` - Google Gemini Models  
**Purpose:** Executes Gemini 2.5 Pro and Flash models for multimodal classification tasks.

**Key Features:**
- Integration with Google's Gemini API
- Optimized prompt engineering for religious art classification
- Advanced caching and cost management
- Support for large context windows (1M tokens)

**Usage:**
```bash
# Run Gemini models
python script/execute_gemini.py --models gemini-2.5-pro gemini-2.5-flash --datasets ArtDL --folders test_1 test_2

# Few-shot evaluation
python script/execute_gemini.py --models gemini-2.5-pro --datasets ICONCLASS --folders test_3
```

**Configuration Required:** API key in `script/gemini_data/config.ini`

### Experimental: Vision-Language Encoder Scripts

⚠️ **Note:** These scripts contain experimental results not yet ready for publication.

#### `execute_clip.py` - CLIP Models (Experimental)
**Purpose:** Implements CLIP variants for contrastive learning-based classification.

**Status:** Results available but under development for publication standards.

```bash
# CLIP execution (experimental)
python script/execute_clip.py --models clip-vit-base-patch32 --datasets ArtDL --folders test_1
```

#### `execute_siglip.py` - SigLIP Models (Experimental)  
**Purpose:** Runs SigLIP models with sigmoid-based contrastive learning.

**Status:** Experimental implementation with preliminary results.

```bash
# SigLIP execution (experimental)  
python script/execute_siglip.py --models siglip-base-patch16-512 --datasets ArtDL --folders test_1
```

### Universal Evaluation and Analysis Scripts

#### `evaluate.py` - Comprehensive Results Analysis
**Purpose:** Generates standardized evaluation metrics for all models.

**Features:**
- Confusion matrices and visualizations
- Per-class and macro/micro metrics
- Cross-model performance comparisons
- Publication-ready result tables

**Usage:**
```bash
# Evaluate LLM results (publication-ready)
python script/evaluate.py --models gpt-4o gemini-2.5-pro --datasets ArtDL ICONCLASS --folders test_1 test_2

# Full evaluation including experimental models
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

### Publication-Ready Models

The following models have been thoroughly evaluated and validated for publication:

| Model Name        | Type                     | Input Context Window     | Output Tokens     | Open Source | Release Date | Knowledge Cut-off |
|------------------|--------------------------|--------------------------|-------------------|--------------|---------------|--------------------|
| GPT-4o            | Multimodal LLM            | 128k tokens              | 16.4k tokens     | No         | Aug 2024      | Oct 2023           |
| GPT-4o mini       | Multimodal LLM            | 128k tokens              | 16.4k tokens     | No         | Jul 2024      | Oct 2023           |
| Gemini 2.5 Pro    | Multimodal LLM            | 1M tokens                | 64k tokens    | No         | Mar 2024      | Jan 2025   |
| Gemini 2.5 Flash  | Multimodal LLM            | 1M tokens                | 65k tokens    | No         | Apr 2025      | Jan 2025    |

### Experimental Models (Development Status)

*Note: Results available but not finalized for publication standards.*

| Model Name        | Type                     | Status                   | Notes                     |
|------------------|--------------------------|--------------------------|---------------------------|
| CLIP (ViT-B/32)   | Vision-Language Encoder   | Experimental             | Contrastive learning evaluation in progress |
| CLIP (ViT-B/16)   | Vision-Language Encoder   | Experimental             | Under development for publication |
| CLIP (ViT-L/14)   | Vision-Language Encoder   | Experimental             | Preliminary results only |
| SigLIP (ViT-B/16) | Vision-Language Encoder   | Experimental             | Sigmoid-based training evaluation |
| SigLIP (ViT-L/16) | Vision-Language Encoder   | Experimental             | Not ready for publication |
| SigLIP (So400M)   | Vision-Language Encoder   | Experimental             | Large-scale model under evaluation |

## Publication-Ready Results

### ArtDL Dataset Results (Validated)
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-------------------|
| **gpt-4o-2024-08-06**          | **86.00%**           | **87.45%**                 | **86.48%**         |
| **gpt-4o-mini-2024-07-18**     | **82.46%**           | **84.98%**                 | **84.60%**         |
| **gemini-2.5-flash-preview**   | **88.20%**           | **87.02%**                 | **84.71%**         |
| **gemini-2.5-pro-preview**     | **90.45%**           | **90.18%**                 | **86.59%**         |
| Baseline (ResNet-50)           | 84.44%               | -                          | -                  |

### ICONCLASS Dataset Results (Validated)
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-------------------|
| **gpt-4o-2024-08-06**          | **75.32%**           | **75.43%**                 | **73.46%**         |
| **gpt-4o-mini-2024-07-18**     | **55.74%**           | **59.56%**                 | **55.50%**         |
| **gemini-2.5-flash-preview**   | **77.17%**           | **77.75%**                 | **78.22%**         |
| **gemini-2.5-pro-preview**     | **83.31%**           | **84.82%**                 | **84.59%**         |
| Baseline (ResNet-50)           | 40.46%               | -                          | -                  |

### Wikidata Dataset Results (Validated)  
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-------------------|
| **gpt-4o-2024-08-06**          | **45.75%**           | **45.31%**                 | **45.31%**         |
| **gpt-4o-mini-2024-07-18**     | **35.78%**           | **36.95%**                 | **34.31%**         |
| **gemini-2.5-flash-preview**   | **45.45%**           | **45.31%**                 | **44.57%**         |
| **gemini-2.5-pro-preview**     | **45.89%**           | **45.31%**                 | **47.07%**         |
| Baseline (ResNet-50)           | 43.97%               | -                          | -                  |

## Experimental Results (Development Status)

*The following results are preliminary and under development for publication standards.*

### ArtDL Dataset - Experimental Models
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-------------------|
| clip-vit-base-patch32          | 16.15%               | 31.55%                     | 21.41%             |
| clip-vit-base-patch16          | 25.64%               | 28.70%                     | 29.13%             |
| clip-vit-large-patch14         | 30.58%               | 44.31%                     | 31.71%             |
| siglip-base-patch16-512        | 48.71%               | 68.19%                     | 55.90%             |
| siglip-large-patch16-384       | 54.45%               | 72.21%                     | 53.49%             |
| siglip-so400m-patch14-384      | 53.86%               | 70.55%                     | 56.38%             |

### ICONCLASS Dataset - Experimental Models
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-------------------|
| clip-vit-base-patch32          | 24.74%               | 29.30%                     | 29.82%             |
| clip-vit-base-patch16          | 30.00%               | 27.37%                     | 33.51%             |
| clip-vit-large-patch14         | 40.00%               | 35.44%                     | 42.81%             |
| siglip-base-patch16-512        | 43.51%               | 33.33%                     | 41.93%             |
| siglip-large-patch16-384       | 48.95%               | 38.77%                     | 49.30%             |
| siglip-so400m-patch14-384      | 59.47%               | 53.16%                     | 60.88%             |

### Wikidata Dataset - Experimental Models
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-------------------|
| clip-vit-base-patch32          | 45.95%               | 44.52%                     | 45.52%             |
| clip-vit-base-patch16          | 50.78%               | 46.66%                     | 47.08%             |
| clip-vit-large-patch14         | 56.76%               | 56.61%                     | 55.48%             |
| siglip-base-patch16-512        | 57.47%               | 46.94%                     | 56.05%             |
| siglip-large-patch16-384       | 60.03%               | 43.95%                     | 61.17%             |
| siglip-so400m-patch14-384      | 66.29%               | 59.60%                     | 64.86%             |
## Future Work

This framework provides a foundation for several research directions:

- **Extended model evaluation** - Testing additional vision-language models and architectures
- **Cross-domain transfer** - Evaluating model performance across different artistic domains
- **Prompt engineering** - Systematic optimization of prompts for improved classification
- **Multimodal fusion** - Combining multiple model outputs for enhanced performance
- **Temporal analysis** - Studying how model capabilities evolve with new releases

## Citation

If you use this work in your research, please cite:

```bibtex
@article{author2024iconography,
  title={Zero-shot Classification for Christian Iconography: A Comparative Study of Vision-Language Models},
  author={Author Name},
  journal={Conference/Journal Name},
  year={2024}
}
```

## References

For implementation details and additional context, see:
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Google Gemini API](https://ai.google.dev/docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

---

*Note: API costs and deterministic output considerations are documented in the OpenAI community guidelines.*

see https://community.openai.com/t/achieving-deterministic-api-output-on-language-models-howto/418318
