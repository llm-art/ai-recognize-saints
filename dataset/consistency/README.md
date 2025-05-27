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

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| gpt-4o | test_1 | 11.76 | 34 | 4 |
| gpt-4o | test_2 | 58.82 | 34 | 20 |
| gpt-4o | test_3 | 67.65 | 34 | 23 |
| gpt-4o-mini | test_1 | 55.88 | 34 | 19 |
| gpt-4o-mini | test_2 | 58.82 | 34 | 20 |
| gpt-4o-mini | test_3 | 50.00 | 34 | 17 |

## Gemini Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| gemini-2.5-flash-preview-04-17 | test_1 | 21.43 | 28 | 6 |
| gemini-2.5-flash-preview-04-17 | test_2 | 30.00 | 30 | 9 |
| gemini-2.5-flash-preview-04-17 | test_3 | 28.12 | 32 | 9 |
| gemini-2.5-pro-preview-05-06 | test_1 | 18.18 | 33 | 6 |
| gemini-2.5-pro-preview-05-06 | test_2 | 15.15 | 33 | 5 |
| gemini-2.5-pro-preview-05-06 | test_3 | 44.44 | 9 | 4 |

## Key Observations

- The clip-vit-large-patch14 model achieves the highest overall consistency (85.29%) on test_2.
- For test_1, the clip-vit-base-patch16 model shows the highest consistency (61.76%).
- For test_2, the clip-vit-large-patch14 model shows the highest consistency (85.29%).
- For test_3, the gpt-4o model shows the highest consistency (67.65%).
- gpt-4o shows extreme variation across tests: ranging from 11.76% to 67.65%.
