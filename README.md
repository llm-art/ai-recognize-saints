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
| Model                     | Test 1   | Test 2   |
|:--------------------------|:---------|:---------|
| clip-vit-base-patch32     | 16.15%   | 31.55%   |
| clip-vit-base-patch16     | 25.64%   | 28.70%   |
| clip-vit-large-patch14    | 30.58%   | 44.31%   |
| gpt-4o-mini               | 85.46%   | 80.00%   |
| siglip-base-patch16-512   | 48.71%   | 68.19%   |
| siglip-large-patch16-384  | 54.45%   | 72.21%   |
| siglip-so400m-patch14-384 | 53.86%   | 70.55%   |
| Baseline                  | 84.44%   | -        |

Test 1 F1 Scores:
| Model                     | Antony of Padua   | Francis of Assisi   | Jerome   | John the Baptist   | Mary Magdalene   | Paul   | Peter   | Saint Dominic   | Saint Sebastian   | Virgin Mary   |
|:--------------------------|:------------------|:--------------------|:---------|:-------------------|:-----------------|:-------|:--------|:----------------|:------------------|:--------------|
| clip-vit-base-patch32     | 0.59%             | 34.61%              | 0.00%    | 19.18%             | 24.32%           | 0.00%  | 0.00%   | 12.67%          | 33.56%            | 10.52%        |
| clip-vit-base-patch16     | 3.33%             | 42.40%              | 0.00%    | 36.80%             | 25.81%           | 0.00%  | 0.00%   | 12.61%          | 40.53%            | 29.17%        |
| clip-vit-large-patch14    | 13.95%            | 64.89%              | 0.00%    | 27.86%             | 30.28%           | 6.35%  | 1.64%   | 16.29%          | 58.54%            | 35.29%        |
| gpt-4o-mini               | 23.53%            | 79.00%              | 74.18%   | 76.61%             | 73.62%           | 64.52% | 80.38%  | 76.19%          | 81.08%            | 93.17%        |
| siglip-base-patch16-512   | 7.07%             | 36.26%              | 4.88%    | 41.49%             | 58.70%           | 0.00%  | 0.00%   | 11.90%          | 65.79%            | 69.30%        |
| siglip-large-patch16-384  | 16.36%            | 53.95%              | 0.00%    | 37.53%             | 58.99%           | 0.00%  | 0.00%   | 14.17%          | 71.76%            | 75.26%        |
| siglip-so400m-patch14-384 | 7.76%             | 46.38%              | 15.38%   | 43.88%             | 66.94%           | 0.00%  | 12.60%  | 25.13%          | 64.63%            | 72.81%        |
| Baseline                  | 64.00%            | 75.35%              | 74.19%   | 67.23%             | 75.58%           | 42.35% | 73.86%  | 56.72%          | 81.19%            | 92.01%        |


Test 2 F1 Scores:
| Model                     | Antony of Padua   | Francis of Assisi   | Jerome   | John the Baptist   | Mary Magdalene   | Paul   | Peter   | Saint Dominic   | Saint Sebastian   | Virgin Mary   |
|:--------------------------|:------------------|:--------------------|:---------|:-------------------|:-----------------|:-------|:--------|:----------------|:------------------|:--------------|
| clip-vit-base-patch32     | 9.21%             | 17.71%              | 51.75%   | 27.59%             | 34.07%           | 22.90% | 22.33%  | 8.51%           | 23.24%            | 41.67%        |
| clip-vit-base-patch16     | 6.09%             | 5.22%               | 51.82%   | 36.76%             | 20.25%           | 30.00% | 32.56%  | 25.81%          | 29.05%            | 36.43%        |
| clip-vit-large-patch14    | 14.43%            | 36.99%              | 59.65%   | 47.78%             | 34.23%           | 32.79% | 46.58%  | 22.80%          | 32.74%            | 54.30%        |
| gpt-4o-mini               | 52.17%            | 59.26%              | 80.00%   | 80.95%             | 82.35%           | 80.00% | 94.12%  | 75.00%          | 60.87%            | 86.59%        |
| siglip-base-patch16-512   | 10.17%            | 33.10%              | 66.67%   | 7.77%              | 40.00%           | 52.17% | 39.79%  | 20.92%          | 44.04%            | 88.15%        |
| siglip-large-patch16-384  | 26.67%            | 53.16%              | 75.11%   | 7.77%              | 37.68%           | 17.54% | 57.67%  | 23.93%          | 58.90%            | 88.78%        |
| siglip-so400m-patch14-384 | 12.50%            | 43.24%              | 80.19%   | 62.59%             | 55.51%           | 54.55% | 62.55%  | 24.49%          | 47.83%            | 84.94%        |
| Baseline                  | 64.00%            | 75.35%              | 74.19%   | 67.23%             | 75.58%           | 42.35% | 73.86%  | 56.72%          | 81.19%            | 92.01%        |

See the [Jupyter Notebook](https://github.com/llm-art/LLM-test/blob/main/cross_evaluate.ipynb)

# References

[1] Milani, Federico, and Piero Fraternali. "A dataset and a convolutional model for iconography classification in paintings." Journal on Computing and Cultural Heritage (JOCCH) 14.4 (2021): 1-18.

[2] E. Maksimova, M.-A. Meimer, M. Piirsalu, P. Järv. Viability of Zero-shot Classification and Search of Historical Photos CHR2024, Aarhus, Denmark, December 4-6, 2024.
