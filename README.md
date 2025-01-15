# Zero-shot classification for Christian iconography

This experiment involves the testing of Zero-shot image classification with LLMs, against the performances of a fine-tuned supervised approach. Specifically, we consider as the baseline the results from [1], reported in Tab. 1. 

Describe [1]...

| Class Name           | # Test Images | Precision | Recall  | F1-Score | Average Precision |
|----------------------|---------------|-----------|---------|----------|-------------------|
| Anthony of Padua     | 14            | 72.73%    | 57.14%  | 64.00%   | 64.14%            |
| Francis of Assisi    | 98            | 69.23%    | 82.65%  | 75.35%   | 76.06%            |
| Jerome               | 118           | 70.77%    | 77.97%  | 74.19%   | 78.88%            |
| John the Baptist     | 99            | 58.09%    | 79.80%  | 67.23%   | 75.69%            |
| Mary Magdalene       | 90            | 79.27%    | 72.22%  | 75.58%   | 82.23%            |
| Paul                 | 52            | 54.55%    | 34.62%  | 42.35%   | 38.47%            |
| Peter                | 119           | 72.95%    | 74.79%  | 73.86%   | 77.93%            |
| Saint Dominic        | 29            | 50.00%    | 65.52%  | 56.72%   | 54.35%            |
| Saint Sebastian      | 56            | 91.11%    | 73.21%  | 81.19%   | 82.46%            |
| Virgin Mary          | 1,189         | 93.04%    | 91.00%  | 92.01%   | 97.03%            |
| **Mean**             |               | 71.17%    | 70.89%  | 70.25%   | 72.73%            |

**Table 1.** Evaluation metrics computed on the test set for paper [1].



## Experiment

This experiment is conducted with the use of the LLM models: CLIP and SigLIP as selected in [2]. Specifically:
* [clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)
* [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
* [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
* [siglip-base-patch16-512](https://huggingface.co/google/siglip-base-patch16-512)
* [siglip-large-patch16-384](https://huggingface.co/google/siglip-large-patch16-384)
* [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)

The test set is downloaded directly from the paper's author official [repository](https://github.com/iFede94/ArtDL/blob/main/sets/test.txt).

Two tests have been done: one where we classify the images with the labels provided by the authors, i.e. the "Label" column in Tab. 2. In the second test, we test the Image Classificator using [IconClass](https://iconclass.org/) description to the label, see Tab. 2, column "Description".

| IconClass ID        | Label               | Description                                                                                                                             |
|---------------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| 11H(ANTONY OF PADUA) | ANTHONY OF PADUA    | the Franciscan monk Antony of Padua; possible attributes: ass, book, crucifix, flowered cross, flaming heart, infant Christ (on book), lily - portrait of male saint |
| 11H(JOHN THE BAPTIST) | JOHN THE BAPTIST    | John the Baptist; possible attributes: book, reed cross, baptismal cup, honeycomb, lamb, staff                                         |
| 11H(PAUL)           | PAUL                | the apostle Paul of Tarsus; possible attributes: book, scroll, sword                                                                   |
| 11H(FRANCIS)        | FRANCIS OF ASSISI   | founder of the Order of Friars Minor (Franciscans), Francis(cus) of Assisi; possible attributes: book, crucifix, lily, skull, stigmata  |
| 11HH(MARY MAGDALENE) | MARY MAGDALENE      | the penitent harlot Mary Magdalene; possible attributes: book (or scroll), crown, crown of thorns, crucifix, jar of ointment, mirror, musical instrument, palm-branch, rosary, scourge |
| 11H(JEROME)         | JEROME              | the monk and hermit Jerome (Hieronymus); possible attributes: book, cardinal's hat, crucifix, hour-glass, lion, skull, stone           |
| 11H(DOMINIC)        | SAINT DOMINIC       | Dominic(us) Guzman of Calerueja, founder of the Order of Preachers (or Dominican (Black) Friars; possible attributes: book, dog with flaming torch, lily, loaf of bread, rosary, star |
| 11F(MARY)           | VIRGIN MARY         | the Virgin Mary                                                                                                                        |
| 11H(PETER)          | PETER               | the apostle Peter, first bishop of Rome; possible attributes: book, cock, (upturned) cross, (triple) crozier, fish, key, scroll, ship, tiara |
| 11H(SEBASTIAN)      | SAINT SEBASTIAN     | the martyr Sebastian; possible attributes: arrow(s), bow, tree-trunk                                                                   

**Table 2.** The classes to test. The labels are provided by the authors. The descriptions are manually retrieved from the IconClass website.

## Evaluations

### Test 1: Labels

For specific model data see [Test 1 dataset](test_1/README.md)


| Model   | Transf. Size | Patch Size | Resolution | ANTHONY OF PADUA | FRANCIS OF ASSISI | JEROME | JOHN THE BAPTIST | MARY MAGDALENE | PAUL  | PETER | SAINT DOMINIC | SAINT SEBASTIAN | VIRGIN MARY |
|---------|--------------|------------|------------|------------------|-------------------|--------|------------------|----------------|-------|-------|---------------|-----------------|-------------|
| CLIP    | base         | 16         | -          | 8.70%           | 39.74%            | 0.00%  | 35.48%           | 24.35%         | 0.00% | 0.00% | 14.73%        | 38.82%          | 29.90%      |
| CLIP    | base         | 32         | -          | 6.20%           | 31.09%            | 0.00%  | 19.30%           | 23.35%         | 0.00% | 0.00% | 10.35%        | 31.25%          | 11.42%      |
| CLIP    | large        | 14         | -          | 11.38%          | 63.51%            | 1.67%  | 27.27%           | 30.40%         | 6.35% | 1.64% | 15.43%        | 58.68%          | 32.98%      |
| SIGLIP  | base         | 16         | 512        | 14.74%          | 43.43%            | 16.11% | 36.24%           | 38.94%         | 0.00% | 1.65% | 11.60%        | 55.62%          | 59.35%      |
| SIGLIP  | large        | 16         | 384        | 16.49%          | 67.02%            | 22.07% | 38.57%           | 55.33%         | 0.00% | 47.18%| 18.12%        | 67.59%          | 72.98%      |
| SIGLIP  | so400m       | 14         | 384        | 7.04%           | 16.51%            | 27.03% | 46.11%           | 57.64%         | 0.00% | 0.00% | 21.40%        | 45.08%          | 71.48%      |

**Table 3.** F1 scores per class

| Model   | Transf. Size | Patch Size | Resolution | mAP (Macro) | mAP (Micro) |
|---------|--------------|------------|------------|-------------|-------------|
| CLIP    | base         | 16         | -          | 34.61%      | 23.14%      |
| CLIP    | base         | 32         | -          | 34.79%      | 20.74%      |
| CLIP    | large        | 14         | -          | 49.06%      | 27.26%      |
| SIGLIP  | base         | 16         | 512        | 48.73%      | 34.80%      |
| SIGLIP  | large        | 16         | 384        | 58.47%      | 39.21%      |
| SIGLIP  | so400m       | 14         | 384        | 62.22%      | 37.48%      |

**Table 4.** mAP Micro and Macro per model

### Test 2: Descriptions

For data regarding specific models see [Test 2 dataset](test_2/README.md)

| Model   | Transf. Size | Patch Size | Resolution | ANTHONY OF PADUA | FRANCIS OF ASSISI | JEROME | JOHN THE BAPTIST | MARY MAGDALENE | PAUL   | PETER  | SAINT DOMINIC | SAINT SEBASTIAN | VIRGIN MARY |
|---------|--------------|------------|------------|------------------|-------------------|--------|------------------|----------------|--------|--------|---------------|-----------------|-------------|
| CLIP    | base         | 16         | -          | 6.12%           | 5.22%            | 51.82% | 36.76%           | 20.15%         | 30.00% | 32.71% | 25.81%        | 29.17%          | 36.43%      |
| CLIP    | base         | 32         | -          | 9.32%           | 17.65%           | 51.75% | 27.75%           | 33.50%         | 22.56% | 22.33% | 8.44%         | 23.24%          | 41.32%      |
| CLIP    | large        | 14         | -          | 14.21%          | 37.21%           | 59.65% | 47.78%           | 34.08%         | 32.52% | 46.58% | 23.04%        | 32.84%          | 54.12%      |
| SIGLIP  | base         | 16         | 512        | 12.83%          | 27.42%           | 54.84% | 5.83%            | 45.14%         | 3.77%  | 54.65% | 0.00%         | 29.27%          | 79.06%      |
| SIGLIP  | large        | 16         | 384        | 21.78%          | 39.42%           | 52.07% | 3.88%            | 49.30%         | 0.00%  | 56.98% | 3.21%         | 57.63%          | 85.82%      |
| SIGLIP  | so400m       | 14         | 384        | 13.79%          | 2.74%            | 64.21% | 68.42%           | 27.06%         | 7.27%  | 63.32% | 3.57%         | 47.62%          | 63.97%      |

**Table 5.** F1 scores per class

| Model   | Transf. Size | Patch Size | Resolution | mAP (Macro) | mAP (Micro) |
|---------|--------------|------------|------------|-------------|-------------|
| CLIP    | base         | 16         | -          | 42.29%      | 28.40%      |
| CLIP    | base         | 32         | -          | 36.41%      | 31.76%      |
| CLIP    | large        | 14         | -          | 54.74%      | 44.17%      |
| SIGLIP  | base         | 16         | 512        | 42.47%      | 42.34%      |
| SIGLIP  | large        | 16         | 384        | 49.57%      | 50.59%      |
| SIGLIP  | so400m       | 14         | 384        | 53.81%      | 42.15%      |

**Table 6.** mAP Micro and Macro per model

# References

[1] Milani, Federico, and Piero Fraternali. "A dataset and a convolutional model for iconography classification in paintings." Journal on Computing and Cultural Heritage (JOCCH) 14.4 (2021): 1-18.

[2] E. Maksimova, M.-A. Meimer, M. Piirsalu, P. Järv. Viability of Zero-shot Classification and Search of Historical Photos CHR2024, Aarhus, Denmark, December 4-6, 2024.
