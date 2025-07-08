# Consistency Analysis Summary

This document provides a summary of the consistency statistics for all models and tests.

Consistency is measured by the percentage of image pairs where both images receive the same prediction.

## CLIP Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| clip-vit-base-patch32 | test_1 | 58.33 | 36 | 21 |
| clip-vit-base-patch32 | test_2 | 75.00 | 36 | 27 |
| clip-vit-base-patch32 | test_3 | 61.11 | 36 | 22 |
| clip-vit-base-patch16 | test_1 | 66.67 | 36 | 24 |
| clip-vit-base-patch16 | test_2 | 75.00 | 36 | 27 |
| clip-vit-base-patch16 | test_3 | 72.22 | 36 | 26 |
| clip-vit-large-patch14 | test_1 | 55.56 | 36 | 20 |
| clip-vit-large-patch14 | test_2 | 86.11 | 36 | 31 |
| clip-vit-large-patch14 | test_3 | 72.22 | 36 | 26 |

## SigLIP Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| siglip-base-patch16-512 | test_1 | 38.89 | 36 | 14 |
| siglip-base-patch16-512 | test_2 | 63.89 | 36 | 23 |
| siglip-base-patch16-512 | test_3 | 33.33 | 36 | 12 |
| siglip-large-patch16-384 | test_1 | 41.67 | 36 | 15 |
| siglip-large-patch16-384 | test_2 | 55.56 | 36 | 20 |
| siglip-large-patch16-384 | test_3 | 41.67 | 36 | 15 |
| siglip-so400m-patch14-384 | test_1 | 55.56 | 36 | 20 |
| siglip-so400m-patch14-384 | test_2 | 77.78 | 36 | 28 |
| siglip-so400m-patch14-384 | test_3 | 61.11 | 36 | 22 |

## GPT Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| gpt-4o-2024-08-06 | test_1 | 25.00 | 36 | 9 |
| gpt-4o-2024-08-06 | test_2 | 27.78 | 36 | 10 |
| gpt-4o-2024-08-06 | test_3 | 27.78 | 36 | 10 |
| gpt-4o-mini-2024-07-18 | test_1 | 22.22 | 36 | 8 |
| gpt-4o-mini-2024-07-18 | test_2 | 30.56 | 36 | 11 |
| gpt-4o-mini-2024-07-18 | test_3 | 25.00 | 36 | 9 |

## Gemini Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| gemini-2.5-flash-preview-05-20 | test_1 | 30.56 | 36 | 11 |
| gemini-2.5-flash-preview-05-20 | test_2 | 30.56 | 36 | 11 |
| gemini-2.5-flash-preview-05-20 | test_3 | 33.33 | 36 | 12 |
| gemini-2.5-pro-preview-05-06 | test_1 | 33.33 | 36 | 12 |
| gemini-2.5-pro-preview-05-06 | test_2 | 33.33 | 36 | 12 |
| gemini-2.5-pro-preview-05-06 | test_3 | 33.33 | 36 | 12 |

## Key Observations

- The clip-vit-large-patch14 model achieves the highest overall consistency (86.11%) on test_2.
- For test_1, the clip-vit-base-patch16 model shows the highest consistency (66.67%).
- For test_2, the clip-vit-large-patch14 model shows the highest consistency (86.11%).
- For test_3, the clip-vit-base-patch16 model shows the highest consistency (72.22%).
- gpt-4o-mini-2024-07-18 shows extreme variation across tests: ranging from 22.22% to 30.56%.
