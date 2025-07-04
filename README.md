# Zero-shot classification for Christian iconography

This experiment involves testing Zero-shot (and few-shot) image classification with LLMs against the performances of a fine-tuned supervised approach.
Specifically, we consider the datasets proposed in [XX, XX, XX].

Primary datasets:

1. [ArtDL](https://artdl.org/)
2. [ICONCLASS AI test set](https://iconclass.org/testset/)
3. [Wikidata]()

For each dataset, we perform three tests:

1. A zero-shot approach with only labels
2. A few-shot approach with 5 to 10 images.
3. zero-shot approach with labels and their descriptions

## Models

In the current experiments, I tested two types of models: Vision-Language Encoder and Multimodal LLMs.

| Model Name        | Type                     | Input Context Window     | Output Tokens     | Open Source | Release Date | Knowledge Cut-off |
|------------------|--------------------------|--------------------------|-------------------|--------------|---------------|--------------------|
| CLIP (ViT-B/32)   | Vision-Language Encoder   | -                      | -               | Yes        | Jan 2021      | -               |
| SIGLIP (ViT-B/16) | Vision-Language Encoder   | -                      | -               | Yes        | Mar 2023      | -               |
| Gemini 2.5 Flash  | Multimodal LLM            | 1M tokens                | 65k tokens    | No         | Apr 2025      | Jan 2025    |
| Gemini 2.5 Pro    | Multimodal LLM            | 1M tokens                | 64k tokens    | No         | Mar 2024      | Jan 2025   |
| GPT-4o            | Multimodal LLM            | 128k tokens              | 16.4k tokens     | No         | Aug 2024      | Oct 2023           |
| GPT-4o mini       | Multimodal LLM            | 128k tokens              | 16.4k tokens     | No         | Jul 2024      | Oct 2023           |

### ArtDL
#### ArtDL Results
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot[*](dataset/ArtDL-data/few-shot/README.md) (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:--------------------------------------------------------------|
| clip-vit-base-patch32          | 16.15%               | 31.55%                     | 21.41%                                                        |
| clip-vit-base-patch16          | 25.64%               | 28.70%                     | 29.13%                                                        |
| clip-vit-large-patch14         | 30.58%               | 44.31%                     | 31.71%                                                        |
| siglip-base-patch16-512        | 48.71%               | 68.19%                     | 55.90%                                                        |
| siglip-large-patch16-384       | 54.45%               | 72.21%                     | 53.49%                                                        |
| siglip-so400m-patch14-384      | 53.86%               | 70.55%                     | 56.38%                                                        |
| gpt-4o-mini                    | 82.78%               | 85.19%                     | 85.41%                                                        |
| gpt-4o                         | 62.30%               | 49.03%                     | 49.22%                                                        |
| gemini-2.5-flash-preview-05-20 | 88.20%               | 87.02%                     | 84.71%                                                        |
| gemini-2.5-pro-preview-05-06   | 90.45%               | 90.18%                     | 86.59%                                                        |
| Baseline                       | 84.44%               |                            |                                                               |

### ICONCLASS
#### ICONCLASS Results
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot[*](dataset/ICONCLASS-data/few-shot/README.md) (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:------------------------------------------------------------------|
| clip-vit-base-patch32          | 24.74%               | 29.30%                     | 29.82%                                                            |
| clip-vit-base-patch16          | 30.00%               | 27.37%                     | 33.51%                                                            |
| clip-vit-large-patch14         | 40.00%               | 35.44%                     | 42.81%                                                            |
| siglip-base-patch16-512        | 43.51%               | 33.33%                     | 41.93%                                                            |
| siglip-large-patch16-384       | 48.95%               | 38.77%                     | 49.30%                                                            |
| siglip-so400m-patch14-384      | 59.47%               | 53.16%                     | 60.88%                                                            |
| gpt-4o-mini                    | 45.13%               | 58.86%                     | 55.97%                                                            |
| gpt-4o                         | 64.04%               | 14.19%                     | 74.16%                                                            |
| gemini-2.5-flash-preview-05-20 | 77.17%               | 77.75%                     | 78.22%                                                            |
| gemini-2.5-pro-preview-05-06   | 83.31%               | 84.82%                     | 84.59%                                                            |
| baseline artdl full            | 12.05%               |                            |                                                                   |
| baseline artdl shared          | 27.96%               |                            |                                                                   |
| baseline resnet50 trained      | 40.46%               |                            |                                                                   |

### Wikidata
#### wikidata Results
| Model                          | zero-shot (labels)   | zero-shot (descriptions)   | few-shot[*](dataset/wikidata-data/few-shot/README.md) (labels)   |
|:-------------------------------|:---------------------|:---------------------------|:-----------------------------------------------------------------|
| clip-vit-base-patch32          | 45.95%               | 44.52%                     | 45.52%                                                           |
| clip-vit-base-patch16          | 50.78%               | 46.66%                     | 47.08%                                                           |
| clip-vit-large-patch14         | 56.76%               | 56.61%                     | 55.48%                                                           |
| siglip-base-patch16-512        | 57.47%               | 46.94%                     | 56.05%                                                           |
| siglip-large-patch16-384       | 60.03%               | 43.95%                     | 61.17%                                                           |
| siglip-so400m-patch14-384      | 66.29%               | 59.60%                     | 64.86%                                                           |
| gpt-4o-mini                    | 35.19%               | 37.10%                     | 34.46%                                                           |
| gpt-4o                         | 44.99%               | 45.52%                     | 43.61%                                                           |
| gemini-2.5-flash-preview-05-20 | 45.45%               | 45.31%                     | 44.57%                                                           |
| gemini-2.5-pro-preview-05-06   | 45.89%               | 45.31%                     | 47.07%                                                           |
| baseline artdl full            | 15.75%               |                            |                                                                  |
| baseline artdl shared          | 28.08%               |                            |                                                                  |
| baseline resnet50 trained      | 43.97%               |                            |                                                                  |

see https://community.openai.com/t/achieving-deterministic-api-output-on-language-models-howto/418318