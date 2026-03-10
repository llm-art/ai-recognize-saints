# Consistency Analysis Summary

This document provides a summary of the consistency statistics for all models and tests.

Consistency is measured by the percentage of image pairs where both images receive the same prediction.

## CLIP Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| clip-vit-base-patch32 | test_1 | 48.89 | 45 | 22 |
| clip-vit-base-patch32 | test_2 | 37.78 | 45 | 17 |
| clip-vit-base-patch32 | test_3 | 42.22 | 45 | 19 |
| clip-vit-base-patch16 | test_1 | 37.78 | 45 | 17 |
| clip-vit-base-patch16 | test_2 | 35.56 | 45 | 16 |
| clip-vit-base-patch16 | test_3 | 42.22 | 45 | 19 |
| clip-vit-large-patch14 | test_1 | 28.89 | 45 | 13 |
| clip-vit-large-patch14 | test_2 | 42.22 | 45 | 19 |
| clip-vit-large-patch14 | test_3 | 35.56 | 45 | 16 |

## SigLIP Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| siglip-base-patch16-512 | test_1 | 35.56 | 45 | 16 |
| siglip-base-patch16-512 | test_2 | 42.22 | 45 | 19 |
| siglip-base-patch16-512 | test_3 | 33.33 | 45 | 15 |
| siglip-large-patch16-384 | test_1 | 31.11 | 45 | 14 |
| siglip-large-patch16-384 | test_2 | 37.78 | 45 | 17 |
| siglip-large-patch16-384 | test_3 | 28.89 | 45 | 13 |
| siglip-so400m-patch14-384 | test_1 | 35.56 | 45 | 16 |
| siglip-so400m-patch14-384 | test_2 | 42.22 | 45 | 19 |
| siglip-so400m-patch14-384 | test_3 | 40.00 | 45 | 18 |

## GPT Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| gpt-4o-2024-08-06 | test_1 | 93.33 | 45 | 42 |
| gpt-4o-2024-08-06 | test_2 | 40.91 | 44 | 18 |
| gpt-4o-2024-08-06 | test_3 | 43.18 | 44 | 19 |
| gpt-4o-mini-2024-07-18 | test_1 | 88.89 | 45 | 40 |
| gpt-4o-mini-2024-07-18 | test_2 | 88.89 | 45 | 40 |
| gpt-4o-mini-2024-07-18 | test_3 | 82.22 | 45 | 37 |
| gpt-5.2-2025-12-11 | test_1 | 84.44 | 45 | 38 |
| gpt-5.2-2025-12-11 | test_2 | 88.89 | 45 | 40 |
| gpt-5.2-2025-12-11 | test_3 | 93.33 | 45 | 42 |
| gpt-5-mini-2025-08-07 | test_1 | 93.33 | 45 | 42 |
| gpt-5-mini-2025-08-07 | test_2 | 91.11 | 45 | 41 |
| gpt-5-mini-2025-08-07 | test_3 | 93.33 | 45 | 42 |

## Gemini Models

| Model | Test | Consistency (%) | Valid Pairs | Same Predictions |
|-------|------|-----------------|-------------|------------------|
| gemini-2.5-flash-preview-05-20 | test_1 | 43.18 | 44 | 19 |
| gemini-2.5-flash-preview-05-20 | test_2 | 45.45 | 44 | 20 |
| gemini-2.5-flash-preview-05-20 | test_3 | 43.18 | 44 | 19 |
| gemini-2.5-pro-preview-05-06 | test_1 | 40.91 | 44 | 18 |
| gemini-2.5-pro-preview-05-06 | test_2 | 43.18 | 44 | 19 |
| gemini-2.5-pro-preview-05-06 | test_3 | 38.64 | 44 | 17 |
| gemini-3-flash-preview | test_1 | 86.67 | 45 | 39 |
| gemini-3-flash-preview | test_2 | 97.78 | 45 | 44 |
| gemini-3-flash-preview | test_3 | 79.55 | 44 | 35 |
| gemini-3.1-pro-preview | test_1 | 100.00 | 45 | 45 |
| gemini-3.1-pro-preview | test_2 | 91.11 | 45 | 41 |
| gemini-3.1-pro-preview | test_3 | 100.00 | 45 | 45 |

## Key Observations

- The gemini-3.1-pro-preview model achieves the highest overall consistency (100.00%) on test_1.
- For test_1, the gemini-3.1-pro-preview model shows the highest consistency (100.00%).
- For test_2, the gemini-3-flash-preview model shows the highest consistency (97.78%).
- For test_3, the gemini-3.1-pro-preview model shows the highest consistency (100.00%).
- gpt-4o-2024-08-06 shows extreme variation across tests: ranging from 40.91% to 93.33%.
