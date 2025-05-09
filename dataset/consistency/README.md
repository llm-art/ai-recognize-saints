# Consistency Analysis Summary

This document provides a summary of the consistency statistics for all models and tests.

Consistency is measured by the percentage of image pairs where both images receive the same prediction.

## CLIP Models

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

## SigLIP Models

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

## GPT Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| gpt-4o | test_1 | 42.42 | 33 | 14 |
| gpt-4o | test_2 | 11.43 | 35 | 4 |
| gpt-4o | test_3 | 88.57 | 35 | 31 |
| gpt-4o-mini | test_1 | 71.43 | 35 | 25 |
| gpt-4o-mini | test_2 | 17.14 | 35 | 6 |
| gpt-4o-mini | test_3 | 57.14 | 35 | 20 |

## Key Observations

- The gpt-4o model achieves the highest overall consistency (88.57%) on test_3.
- For test_1, the gpt-4o-mini model shows the highest consistency (71.43%).
- For test_2, the siglip-so400m-patch14-384 model shows the highest consistency (80.00%).
- For test_3, the gpt-4o model shows the highest consistency (88.57%).
- gpt-4o shows extreme variation across tests: ranging from 11.43% to 88.57%.
