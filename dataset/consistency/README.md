# Consistency Analysis Summary

This document provides a summary of the consistency statistics for all models and tests.

Consistency is measured by the percentage of image pairs where both images receive the same prediction.

## CLIP Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| clip-vit-base-patch32 | test_1 | 55.88 | 34 | 19 |
| clip-vit-base-patch32 | test_2 | 73.53 | 34 | 25 |
| clip-vit-base-patch32 | test_3 | 61.76 | 34 | 21 |
| clip-vit-base-patch16 | test_1 | 61.76 | 34 | 21 |
| clip-vit-base-patch16 | test_2 | 70.59 | 34 | 24 |
| clip-vit-base-patch16 | test_3 | 64.71 | 34 | 22 |
| clip-vit-large-patch14 | test_1 | 52.94 | 34 | 18 |
| clip-vit-large-patch14 | test_2 | 85.29 | 34 | 29 |
| clip-vit-large-patch14 | test_3 | 64.71 | 34 | 22 |

## SigLIP Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| siglip-base-patch16-512 | test_1 | 41.18 | 34 | 14 |
| siglip-base-patch16-512 | test_2 | 67.65 | 34 | 23 |
| siglip-base-patch16-512 | test_3 | 35.29 | 34 | 12 |
| siglip-large-patch16-384 | test_1 | 44.12 | 34 | 15 |
| siglip-large-patch16-384 | test_2 | 55.88 | 34 | 19 |
| siglip-large-patch16-384 | test_3 | 44.12 | 34 | 15 |
| siglip-so400m-patch14-384 | test_1 | 58.82 | 34 | 20 |
| siglip-so400m-patch14-384 | test_2 | 79.41 | 34 | 27 |
| siglip-so400m-patch14-384 | test_3 | 61.76 | 34 | 21 |

## GPT Models

## GEMINI Models


## Key Observations

- The clip-vit-large-patch14 model achieves the highest overall consistency (85.29%) on test_2.
- For test_1, the clip-vit-base-patch16 model shows the highest consistency (61.76%).
- For test_2, the clip-vit-large-patch14 model shows the highest consistency (85.29%).
- For test_3, the clip-vit-base-patch16 model shows the highest consistency (64.71%).
- gpt-4o-mini shows extreme variation across tests: ranging from 50.00% to 61.76%.
