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
| Model                          | zero-shot (labels)   | few-shot[*](dataset/ArtDL-data/few-shot/README.md) (labels)   | zero-shot (descriptions)   |
|:-------------------------------|:---------------------|:--------------------------------------------------------------|:---------------------------|
| clip-vit-base-patch32          | 16.15%               | 21.41%                                                        | 31.55%                     |
| clip-vit-base-patch16          | 25.64%               | 29.13%                                                        | 28.70%                     |
| clip-vit-large-patch14         | 30.58%               | 31.71%                                                        | 44.31%                     |
| gemini-2.5-flash-preview-04-17 | 47.57%               | 48.85%                                                        | 49.14%                     |
| gemini-2.5-pro-preview-05-06   | 50.54%               | 58.72%                                                        | 66.76%                     |
| siglip-base-patch16-512        | 48.71%               | 55.90%                                                        | 68.19%                     |
| siglip-large-patch16-384       | 54.45%               | 53.49%                                                        | 72.21%                     |
| siglip-so400m-patch14-384      | 53.86%               | 56.38%                                                        | 70.55%                     |
| gpt-4o-mini                    | 75.32%               | 72.62%                                                        | 78.76%                     |
| gpt-4o                         | 64.20%               | 85.68%                                                        | 87.02%                     |
| Baseline                       | 84.44%               |                                                               |                            |

### ICONCLASS

#### ICONCLASS Results
| Model                          | zero-shot (labels)   | few-shot[*](dataset/ICONCLASS-data/few-shot/README.md) (labels)   | zero-shot (descriptions)   |
|:-------------------------------|:---------------------|:------------------------------------------------------------------|:---------------------------|
| clip-vit-base-patch32          | 23.48%               | 29.56%                                                            | 26.35%                     |
| clip-vit-base-patch16          | 28.89%               | 32.60%                                                            | 26.35%                     |
| clip-vit-large-patch14         | 39.36%               | 42.74%                                                            | 34.29%                     |
| gemini-2.5-flash-preview-04-17 | 15.28%               | 14.67%                                                            | 17.79%                     |
| siglip-base-patch16-512        | 43.41%               | 42.23%                                                            | 31.59%                     |
| siglip-large-patch16-384       | 49.16%               | 49.49%                                                            | 36.82%                     |
| siglip-so400m-patch14-384      | 57.77%               | 59.46%                                                            | 51.69%                     |
| gpt-4o-mini                    | 53.21%               | 52.53%                                                            | 22.58%                     |
| gpt-4o                         | 65.03%               | 68.41%                                                            | 68.92%                     |

### Wikidata

#### wikidata Results
| Model                          | zero-shot (labels)   | few-shot[*](dataset/wikidata-data/few-shot/README.md) (labels)   | zero-shot (descriptions)   |
|:-------------------------------|:---------------------|:-----------------------------------------------------------------|:---------------------------|
| clip-vit-base-patch32          | 45.95%               | 45.52%                                                           | 44.52%                     |
| clip-vit-base-patch16          | 50.78%               | 47.08%                                                           | 46.66%                     |
| clip-vit-large-patch14         | 56.76%               | 55.48%                                                           | 56.61%                     |
| gemini-2.5-flash-preview-04-17 | 58.87%               | 48.70%                                                           | 52.06%                     |
| siglip-base-patch16-512        | 57.47%               | 56.05%                                                           | 46.94%                     |
| siglip-large-patch16-384       | 60.03%               | 61.17%                                                           | 43.95%                     |
| siglip-so400m-patch14-384      | 66.29%               | 64.86%                                                           | 59.60%                     |
| gpt-4o-mini                    | 48.58%               | 48.29%                                                           | 51.14%                     |
| gpt-4o                         | 50.87%               | 65.19%                                                           | 57.41%                     |
