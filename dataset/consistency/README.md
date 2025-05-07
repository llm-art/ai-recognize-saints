# Consistency Analysis Data

This directory contains consistency data for different models and tests. The data is organized in the following structure:

```
dataset/consistency/
├── {model}/
│   ├── {test}/
│   │   └── consistency_data.json
├── example/
│   ├── correct_1/
│   │   ├── image_1_*.jpg
│   │   └── image_2_*.jpg
│   ├── incorrect_1/
│   │   ├── image_1_*.jpg
│   │   └── image_2_*.jpg
│   └── ...
└── examples.md
```

Where:
- `{model}` is one of the following:
  - CLIP models: clip-vit-base-patch32, clip-vit-base-patch16, clip-vit-large-patch14
  - SigLIP models: siglip-base-patch16-512, siglip-large-patch16-384, siglip-so400m-patch14-384
  - GPT models: gpt-4o, gpt-4o-mini
- `{test}` is one of: test_1, test_2, test_3

## Data Format

Each `consistency_data.json` file contains an array of image pairs, where each pair consists of two images from different datasets that are visually similar. For each image, the following information is provided:

```json
{
  "dataset": "Dataset name (e.g., ArtDL, ICONCLASS, wikidata)",
  "name": "Image name",
  "path": "Path to the image file",
  "ground_truth": "Ground truth class label",
  "predicted": "Predicted class label by the model"
}
```

## Consistency Statistics

The consistency of a model is measured by the percentage of image pairs where both images receive the same prediction. Here are the consistency statistics for each model and test:

### CLIP Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| clip-vit-base-patch32 | test_1 | 40.00 | 35 | 14 |
| clip-vit-base-patch32 | test_2 | 62.86 | 35 | 22 |
| clip-vit-base-patch32 | test_3 | 48.57 | 35 | 17 |
| clip-vit-base-patch16 | test_1 | 51.43 | 35 | 18 |
| clip-vit-base-patch16 | test_2 | 60.00 | 35 | 21 |
| clip-vit-base-patch16 | test_3 | 65.71 | 35 | 23 |
| clip-vit-large-patch14 | test_1 | 48.57 | 35 | 17 |
| clip-vit-large-patch14 | test_2 | 62.86 | 35 | 22 |
| clip-vit-large-patch14 | test_3 | 60.00 | 35 | 21 |

### SigLIP Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| siglip-base-patch16-512 | test_1 | 31.43 | 35 | 11 |
| siglip-base-patch16-512 | test_2 | 65.71 | 35 | 23 |
| siglip-base-patch16-512 | test_3 | 28.57 | 35 | 10 |
| siglip-large-patch16-384 | test_1 | 40.00 | 35 | 14 |
| siglip-large-patch16-384 | test_2 | 57.14 | 35 | 20 |
| siglip-large-patch16-384 | test_3 | 40.00 | 35 | 14 |
| siglip-so400m-patch14-384 | test_1 | 57.14 | 35 | 20 |
| siglip-so400m-patch14-384 | test_2 | 80.00 | 35 | 28 |
| siglip-so400m-patch14-384 | test_3 | 57.14 | 35 | 20 |

### GPT Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| gpt-4o | test_1 | 42.42 | 33 | 14 |
| gpt-4o | test_2 | 11.43 | 35 | 4 |
| gpt-4o | test_3 | 88.57 | 35 | 31 |
| gpt-4o-mini | test_1 | 71.43 | 35 | 25 |
| gpt-4o-mini | test_2 | 17.14 | 35 | 6 |
| gpt-4o-mini | test_3 | 57.14 | 35 | 20 |

## Generating the Data

The data was generated using the `generate_consistency_data.py` script, which:

1. Loads image pairs from `robust_cross_duplicates.json`
2. For each model and test:
   - Creates the directory structure
   - Loads ground truth data from the appropriate dataset's ground truth file
   - Gets model predictions for each image
   - Creates a JSON file with the original structure plus ground_truth and predicted fields

To regenerate the data, run:

```bash
cd LLM-test
python script/generate_consistency_data.py
```

## Example Image Pairs

The `example` directory contains selected image pairs that demonstrate correct and incorrect predictions by the gpt-4o model on test_3 (which has the highest consistency score). These examples help visualize what types of images the model handles well and where it struggles.

To generate these examples and the accompanying markdown table, run:

```bash
cd LLM-test
python script/generate_examples.py
```

This will:
1. Select 5 correctly predicted and 5 incorrectly predicted image pairs
2. Copy these images to the `dataset/consistency/example` directory
3. Generate a markdown table in `dataset/consistency/examples.md`

For a detailed view of these examples, see [examples.md](examples.md).

## Key Observations

- The gpt-4o model achieves the highest overall consistency (88.57%) on test_3, significantly outperforming other models.
- The siglip-so400m-patch14-384 model achieves the highest consistency (80%) on test_2.
- GPT models show extreme variation across tests: gpt-4o ranges from 11.43% (test_2) to 88.57% (test_3).
- The gpt-4o-mini model shows strong consistency (71.43%) on test_1, outperforming all CLIP and SigLIP models on this test.
- Test_2 generally yields higher consistency scores for CLIP and SigLIP models, but significantly lower scores for GPT models.
- Test_3 appears to be the most favorable for GPT models, with gpt-4o achieving its highest consistency score.
