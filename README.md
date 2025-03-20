# Zero-shot classification for Christian iconography

This experiment involves testing Zero-shot (and few-shot) image classification with LLMs against the performances of a fine-tuned supervised approach.
Specifically, we consider the datasets proposed in [XX, XX, XX].

Primary datasets:

1. [ArtDL](https://artdl.org/)
2. [WikiArt](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset)

Secondary datasets:
1. [IconArt](https://zenodo.org/records/4737435)
2. [IICONGRAPH](https://zenodo.org/records/10294589)

For each dataset, we perform three tests:

1. A zero-shot approach with only labels
2. A few-shot approach with 5 to 10 images.
3. zero-shot approach with labels and their descriptions


## Experiment

This experiment is conducted with the use of the LLM models: CLIP and SigLIP as selected in [2]. Specifically:
* [clip-vit-base-patch16](https://huggingface.co/openai/clip-vit-base-patch16)
* [clip-vit-base-patch32](https://huggingface.co/openai/clip-vit-base-patch32)
* [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
* [siglip-base-patch16-512](https://huggingface.co/google/siglip-base-patch16-512)
* [siglip-large-patch16-384](https://huggingface.co/google/siglip-large-patch16-384)
* [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)
* [gpt-4o-mini](https://pypi.org/project/openai/)
* [gpt-4o](https://pypi.org/project/openai/)

## Datasets

### ArtDL

ArtDL is a comprehensive dataset designed for iconography classification in paintings, primarily from the Renaissance period, focusing on Christian art. It comprises 42,479 images sourced from 10 online museums and open data collections. Each painting is annotated into one of 19 classes based on the Iconclass classification system, which is widely used for art and iconography research.

The test set is downloaded directly from the paper's author's official [repository](https://github.com/iFede94/ArtDL/blob/main/sets/test.txt).

Two tests have been done: one where we classify the images with the labels provided by the authors, i.e. the "Label" column in Tab. 2. In the second test, we test the Image Classificator using [IconClass](https://iconclass.org/) description to the label, see Tab. 2, column "Description".

| IconClass ID        | Label               | Description                                                                                                                             |
|---------------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| 11H(ANTONY OF PADUA) | Antony of Padua   | the Franciscan monk Antony of Padua; possible attributes: ass, book, crucifix, flowered cross, flaming heart, infant Christ (on book), lily - portrait of male saint |
| 11H(JOHN THE BAPTIST) | John the Baptist    | John the Baptist; possible attributes: book, reed cross, baptismal cup, honeycomb, lamb, staff                                         |
| 11H(PAUL)           | Paul                | the apostle Paul of Tarsus; possible attributes: book, scroll, sword                                                                   |
| 11H(FRANCIS)        | Francis of Assisi   | founder of the Order of Friars Minor (Franciscans), Francis(cus) of Assisi; possible attributes: book, crucifix, lily, skull, stigmata  |
| 11HH(MARY MAGDALENE) | Mary Magdalene      | the penitent harlot Mary Magdalene; possible attributes: book (or scroll), crown, crown of thorns, crucifix, jar of ointment, mirror, musical instrument, palm-branch, rosary, scourge |
| 11H(JEROME)         | Jerome              | the monk and hermit Jerome (Hieronymus); possible attributes: book, cardinal's hat, crucifix, hour-glass, lion, skull, stone           |
| 11H(DOMINIC)        | Saint Dominic       | Dominic(us) Guzman of Calerueja, founder of the Order of Preachers (or Dominican (Black) Friars; possible attributes: book, dog with flaming torch, lily, loaf of bread, rosary, star |
| 11F(MARY)           | Virgin Mary         | the Virgin Mary                                                                                                                        |
| 11H(PETER)          | Peter               | the apostle Peter, first bishop of Rome; possible attributes: book, cock, (upturned) cross, (triple) crozier, fish, key, scroll, ship, tiara |
| 11H(SEBASTIAN)      | Saint Sebastian     | the martyr Sebastian; possible attributes: arrow(s), bow, tree-trunk                                                                   

**Table X.** The classes to test. The labels are provided by the authors. The descriptions are manually retrieved from the IconClass website.

#### Results

Accuracy per model

| Model                     | zero-shot (labels)   | few-shot[*](dataset/ArtDL-data/README.md) (labels)   | zero-shot (descriptions)   |
|:--------------------------|:---------------------|:-----------------------------------------------------|:---------------------------|
| clip-vit-base-patch32     | 16.15%               | 21.41%                                               | 31.55%                     |
| clip-vit-base-patch16     | 25.64%               | 29.13%                                               | 28.70%                     |
| clip-vit-large-patch14    | 30.58%               | 31.71%                                               | 44.31%                     |
| siglip-base-patch16-512   | 48.71%               | 55.90%                                               | 68.19%                     |
| siglip-large-patch16-384  | 54.45%               | 53.49%                                               | 72.21%                     |
| siglip-so400m-patch14-384 | 53.86%               | 56.38%                                               | 70.55%                     |
| gpt-4o-mini               | 69.96%               | 82.46%                                               | 63.68%                     |
| gpt-4o                    | 85.03%               | 85.03%                                               | 89.06%                     |
| Baseline                  | 84.44%               | -                                                    | -                          |


**Table X.**



# References

[1] Milani, Federico, and Piero Fraternali. "A dataset and a convolutional model for iconography classification in paintings." Journal on Computing and Cultural Heritage (JOCCH) 14.4 (2021): 1-18.

[2] ...

[3] E. Maksimova, M.-A. Meimer, M. Piirsalu, P. Järv. Viability of Zero-shot Classification and Search of Historical Photos CHR2024, Aarhus, Denmark, December 4-6, 2024.

[4] N. Gonthier, «IconArt Dataset». Zenodo, ott. 05, 2018. doi: 10.5281/zenodo.4737435.
