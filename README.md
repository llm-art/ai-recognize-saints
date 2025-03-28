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

| Name             | Model Type   | Architecture            | Vision Component       | Text Component         | Training Objective                    | Loss Function          | Parameter Count | Token Limit | Image Resolution | Training Data Size     | License         |
|------------------|--------------|--------------------------|------------------------|------------------------|----------------------------------------|-------------------------|------------------|--------------|-------------------|------------------------|------------------|
| [BLIP](https://huggingface.co/docs/transformers/model_doc/blip)             | VLM          | Dual Encoder + Decoder   | ViT-B/L                | BERT-style or T5       | Captioning + VQA                      | Contrastive + CE       | ~250M–500M       | 512           | 224×224           | COCO + web data         | Salesforce (open)|
| [BLIP-2](https://huggingface.co/docs/transformers/model_doc/blip-2)           | VLM          | Vision-Language Bridge   | ViT-G/14 or ViT-L      | Frozen LLM (OPT/T5)    | Pretrain + Q-former + Instruction     | Contrastive + CE       | ~1.2B            | 512–1024      | 224×224+           | Public mixes            | Salesforce (open)|
| BLIP-Tiny        | VLM          | Efficient Multimodal     | ViT-Tiny               | Tiny BERT or T5        | Few-shot, distilled instruction       | Cross-Entropy          | <100M            | 128–256       | 224×224           | COCO + web data         | Salesforce (open)|
| [CLIP (ViT-B/32)](https://huggingface.co/openai/clip-vit-base-patch32)  | VLM          | Dual Encoder             | ViT-B/32               | Transformer (GPT-like) | Contrastive Learning                   | InfoNCE (Softmax)       | ~125M            | 77           | 224×224           | 400M pairs              | OpenAI (open)    |
| [Gemini](https://huggingface.co/describeai/gemini)           | LMM          | Unified Transformer      | Internal vision encoder| Gemini Text Decoder    | Multimodal Instruction Following      | Cross-Entropy          | 770M | ~128k         | Variable           | Proprietary             | Google (closed)  |
| [GIT](https://huggingface.co/docs/transformers/model_doc/git) | VLM          | Unified Transformer      | CNN or ViT             | T5-style decoder       | Captioning, VQA, Dense prediction     | Cross-Entropy          | ~750M            | 512           | 224×224           | Internal + COCO         | Google (open)    |
| [GPT-4o](https://platform.openai.com/docs/models/gpt-4o) | LMM          | Unified Transformer      | Internal vision module | GPT-4o Text Decoder    | Instruction Following + Multimodal    | Cross-Entropy          | Not disclosed    | ~128k        | Variable           | Proprietary             | OpenAI (closed)  |
| [GPT-4o Mini](https://platform.openai.com/docs/models/gpt-4o-mini)      | LMM          | Unified Transformer      | Internal vision module | GPT-4o Mini Text       | Instruction Following + Multimodal    | Cross-Entropy          | Not disclosed    | ~32k         | Variable           | Proprietary             | OpenAI (closed)  |
| [ImageBind](https://imagebind.metademolab.com/) | Multimodal   | Cross-modal Embedding    | ViT-L                  | BERT-style             | Joint embedding across modalities     | Contrastive            | ~500M+           | 512           | 224×224           | Audio+Text+Image        | Meta (open)      |
| [Kosmos-2](https://huggingface.co/docs/transformers/model_doc/kosmos-2) | VLM          | Unified Transformer      | Internal visual encoder| Transformer Decoder    | Multimodal Pretraining + VQA          | Cross-Entropy          | ~1.6B–2B         | 2048          | 224×224+           | Multimodal web data     | Microsoft (open) |
| [LLaVA (13B)](https://huggingface.co/docs/transformers/model_doc/llava) | VLM          | Vision-Language Bridge   | CLIP-ViT-L/336px       | Vicuna 13B (LLaMA)     | Instruction Tuning (VQA, GPT4Gen)     | Cross-Entropy          | ~13B             | 2048          | 336×336           | ~1.2M                  | Academic (open)  |
| [MiniGPT-4](https://minigpt-4.github.io/) | VLM          | Vision-Language Bridge   | CLIP-ViT-L/14          | Vicuna 7B (LLaMA)      | Instruction Tuning + Captioning       | Cross-Entropy          | ~7B              | 2048          | 224–336px         | ~3M + GPT4 responses    | Academic (open)  |
| [SEED](https://ailab-cvc.github.io/seed/index.html) | Multimodal   | Cross-modal Embedding    | ViT-variant            | BERT-style             | Semantic Alignment                    | Contrastive            | Experimental     | 512           | 224×224           | Multisensory            | Meta (open)      |
| [SigLIP (ViT-L/14)]([https://huggingface.co/google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-large-patch16-256))| VLM          | Dual Encoder             | ViT-L/14               | BERT-style Transformer | Contrastive Learning                   | Sigmoid                | ~478M            | 128–256       | 224–384px         | 1B+ pairs               | Google (open)    |


### Model Type

* **[VLM (Vision-Language Model)](https://huggingface.co/blog/vlms)**: Models that process both visual and textual data for tasks like captioning, VQA, and classification.
* **[LMM (Large Multimodal Model)](https://research.aimultiple.com/large-multimodal-models/)**: General-purpose models that combine vision, language, and often other modalities (audio, video) for instruction following and generation.
* **Multimodal**: Broader models trained across multiple sensory modalities (e.g., image, audio, text) and meant for embedding or alignment.

### Architecture

* **Dual Encoder:** Independent vision and text encoders projected into a shared embedding space (e.g., CLIP, SigLIP).
* **Unified Transformer**: A single transformer model that processes both visual and textual inputs (e.g., Kosmos-2, Gemini).
* **Vision-Language Bridge:** A connector module (MLP, Q-former) bridges vision encoder to a language model (e.g., BLIP-2, LLaVA).
* **Cross-modal Embedding:** Aligns multimodal embeddings into a shared semantic space (e.g., ImageBind, SEED).

### Vision Component

* **Backbone used to encode image data:** ViT-B/L, CLIP-ViT, CNN, or internal modules in closed-source models.
* **LLM or encoder used for language:** Transformer (GPT-like), BERT-style, T5-style, or frozen models.

### Training Objective

* **Contrastive Learning:** Match correct image-text pairs.
* **Captioning:** Generate descriptive text for images.
* **Instruction Following:** Respond to prompts in few-shot style.
* **VQA:** Answer questions based on image context.
* **Joint Embedding:** Learn aligned representations across modalities.

### Loss Function

* **InfoNCE (Softmax):** Contrastive loss using softmax over similarities.
* **Sigmoid:** Binary prediction for image-text match.
* **Cross-Entropy (CE):** Standard loss for classification/generation.
* **Contrastive + CE:** Hybrid training for matching and generation.


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

| Dataset name | Task | Images | Metadata | Availability | License | Last update |
|--------------|------|--------|----------|--------------|---------|-------------|
| [Art 500k](https://deepart.hkust.edu.hk/ART500K/art500k.html) | Classification | ~500K | 10 attributes (artist, genre, art movement, etc) | Yes | Non-commercial research only | 2019 |
| [ArtDL](https://artdl.org/) | Classification | ~42.5K | 10 classes of Christian Icons | Yes | Academic research only | 2021 |
| [DEArt](https://b2share.eudat.eu/records/449856a0b31543fc81baf19834f37d9d) | Classification | ~40K | Artist, title, year, subject (Iconclass), etc. | Yes | CC BY 4.0 | 2021 |
| [Iconclass AI Test Set](https://iconclass.org/testset/) | Classification | ~87.5K | Iconclass IDs | Yes | Open license via Arkyves and partners | 2020 |
| [IconArt](https://zenodo.org/records/4737435) | Object detection | ~6K | Up to 10 iconographic classes | Yes | CC BY 4.0 | 2018 |
| [IICONGRAPH](https://zenodo.org/records/10294589) | Classification | 3,330 | Annotated iconographic classes (Christian saints & scenes) | Yes | CC BY 4.0 | 2024 |
| [MET dataset](https://cmp.felk.cvut.cz/met/) | Classification | ~400K | Title, artist, date, material, tags | Yes | Research use (license unclear) ❓ | 2020 |
| [OMNIart](https://github.com/dboxdt/omniart) | Classification, Retrieval | ~1.5M | Artist, title, creation date, type, etc. | Yes | CC licenses vary | 2018 |
| [Painting-91](https://people.eecs.berkeley.edu/~jiayq/painting/) | Classification | 4,266 | 91 style categories | Yes | Research use only ❓ | 2012 |
| [Portrait Painting Dataset for Different Movements](https://www.kaggle.com/datasets/sooryaprakash/portrait-painting-dataset-for-different-movement) | Classification | ~1,100 | Labels of art movement | Yes | Unknown / Kaggle dataset ❓ | 2020 |
| [Rijksmuseum 2014](https://www.kaggle.com/c/rijksmuseum-challenge/data) | Classification | 112K | Title, artist, object type, etc. | Yes | Research purposes only | 2014 |
| [WGA (Web Gallery of Art)](https://www.wga.hu/) | Classification | ~50K | Artist, title, date, school, etc. | Partial | No clear license / scraping needed ❓ | Ongoing |
| [WikiArt](https://www.wikiart.org/) | Classification | ~250K | Artist, genre, style, etc. | No (corrupted zip, dead links) | N/A | Last known update: 2016 |
| [Wikidata (via Iconclass or SPARQL queries)](https://www.wikidata.org/) | Classification | Varies | ICONCLASS IDs, metadata via structured queries | Yes | CC0 | Ongoing |

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

### ICONCLASS AI test set

A test dataset and challenge to apply machine learning to collections described with the Iconclass classification system. The json file is a map of filenames to Iconclass notations, here is what the first few entries look like:
```
{
  "IIHIM_1956438510.jpg": [
    "31A235",
    "31A24(+1)",
    "61B(+54)",
    "31A2212(+1)",
    "31D14"
  ],
  "IIHIM_-859728949.jpg": [
    "41D92",
    "25G41"
  ],
  "IIHIM_1207680098.jpg": [
    "11H",
    "11I35",
    "11I36"
  ],
  "IIHIM_-743518586.jpg": [
    "11F25",
    "11FF25",
    "41E2"
  ]
}
```
The current dataset tested has 592 single-labelled images with only Male and Female Saints (starting with 11H or 11HH)
```
                       ID  ImageCount
0               11H(PAUL)         178
1             11H(JEROME)         158
2    11HH(MARY MAGDALENE)         153
3               11H(JOHN)         132
4              11H(PETER)         128
5         11HH(CATHERINE)         116
6       11H(ANTONY ABBOT)         109
7            11H(MATTHEW)          94
8            11H(FRANCIS)          78
9               11H(MARK)          73
10  11H(JOHN THE BAPTIST)          67
```

The labels have been taken from the first ICONCLASS subclass with name e.g., 11H(PAUL)1 

```
 11H(PAUL)1 specific aspects ~ St. Paul
 ```

| ID                     | Label                 | Description                                                                                               |
|------------------------|----------------------|-----------------------------------------------------------------------------------------------------------|
| 11H(PAUL)             | St. Paul             | the apostle Paul of Tarsus; possible attributes: book, scroll, sword                                      |
| 11H(JEROME)           | St. Jerome           | the monk and hermit Jerome (Hieronymus); possible attributes: book, cardinal's hat, crucifix, hour-glass, lion, skull, stone |
| 11HH(MARY MAGDALENE)  | Mary Magdalene       | the penitent harlot Mary Magdalene; possible attributes: book (or scroll), crown, crown of thorns, crucifix, jar of ointment, mirror, musical instrument, palm-branch, rosary, scourge |
| 11H(JOHN)            | St. John the Evangelist | the apostle John the Evangelist; possible attributes: book, cauldron, chalice with snake, eagle, palm, scroll |
| 11H(PETER)           | St. Peter            | the apostle Peter, first bishop of Rome; possible attributes: book, cock, (upturned) cross, (triple) crozier, fish, key, scroll, ship, tiara |
| 11HH(CATHERINE)      | St. Catherine        | the virgin martyr Catherine of Alexandria; possible attributes: book, crown, emperor Maxentius, palm-branch, ring, sword, wheel |
| 11H(ANTONY ABBOT)   | St. Anthony Abbot    | the hermit Antony Abbot (Antonius Abbas) of Egypt, also called the Great; possible attributes: bell, book, T-shaped staff, flames, pig |
| 11H(MATTHEW)        | St. Matthew         | the apostle and evangelist Matthew (Mattheus); possible attributes: angel, axe, book, halberd, pen and inkhorn, purse, scroll, square, sword |
| 11H(FRANCIS)        | St. Francis of Assisi | founder of the Order of Friars Minor (Franciscans), Francis(cus) of Assisi; possible attributes: book, crucifix, lily, skull, stigmata |
| 11H(MARK)          | St. Mark             | Mark (Marcus) the evangelist, and bishop of Alexandria; possible attributes: book, (winged) lion, pen and inkhorn, scroll |
| 11H(JOHN THE BAPTIST) | St. John the Baptist | John the Baptist; possible attributes: book, reed cross, baptismal cup, honeycomb, lamb, staff |

#### Results

| Model                     | zero-shot (labels)   | few-shot[*](dataset/ICONCLASS-data/few-shot/README.md) (labels)   | zero-shot (descriptions)   |
|:--------------------------|:---------------------|:------------------------------------------------------------------|:---------------------------|
| clip-vit-base-patch32     | 23.48%               | 29.56%                                                            | 26.35%                     |
| clip-vit-base-patch16     | 28.89%               | 32.60%                                                            | 26.35%                     |
| clip-vit-large-patch14    | 39.36%               | 42.74%                                                            | 34.29%                     |
| siglip-base-patch16-512   | 43.41%               | 42.23%                                                            | 31.59%                     |
| siglip-large-patch16-384  | 49.16%               | 49.49%                                                            | 36.82%                     |
| siglip-so400m-patch14-384 | 57.77%               | 59.46%                                                            | 51.69%                     |
| gpt-4o-mini               | 50.17%               | 49.16%                                                            | 52.20%                     |
| gpt-4o                    | 54.86%               | 65.88%                                                            | 34.07%                     |


### Wikidata

With the following SPARQL query I downloaded the paintings, from Wikidata, with at least one ICONCLASS class related, starting with 11H or 11HH

```
SELECT ?painting ?image ?iconclass WHERE {
  ?painting wdt:P31 wd:Q3305213;        
           wdt:P1257 ?iconclass.        
  ?painting wdt:P18 ?image.            
  FILTER(strstarts(?iconclass, '11H'))
}
```

The result, for paintings with single label, is 724 images with these classes:

```
11HH(MARY MAGDALENE)     177
11H(JOHN THE BAPTIST)    131
11H(JEROME)               78
11HH(CATHERINE)           76
11H(PETER)                68
11H(JOHN)                 51
11H(FRANCIS)              40
11H(ANTONY ABBOT)         38
11H(JOSEPH)               35
11H(PAUL)                 31
```

With these data:
| ID                     | Label                 | Description                                                                                               |
|------------------------|----------------------|-----------------------------------------------------------------------------------------------------------|
| 11HH(MARY MAGDALENE)  | Mary Magdalene       | the penitent harlot Mary Magdalene; possible attributes: book (or scroll), crown, crown of thorns, crucifix, jar of ointment, mirror, musical instrument, palm-branch, rosary, scourge |
| 11H(JOHN THE BAPTIST) | St. John the Baptist | John the Baptist; possible attributes: book, reed cross, baptismal cup, honeycomb, lamb, staff            |
| 11H(JEROME)           | St. Jerome           | the monk and hermit Jerome (Hieronymus); possible attributes: book, cardinal's hat, crucifix, hour-glass, lion, skull, stone |
| 11HH(CATHERINE)       | St. Catherine        | the virgin martyr Catherine of Alexandria; possible attributes: book, crown, emperor Maxentius, palm-branch, ring, sword, wheel |
| 11H(PETER)            | St. Peter            | the apostle Peter, first bishop of Rome; possible attributes: book, cock, (upturned) cross, (triple) crozier, fish, key, scroll, ship, tiara |
| 11H(JOHN)             | St. John the Evangelist | the apostle John the Evangelist; possible attributes: book, cauldron, chalice with snake, eagle, palm, scroll |
| 11H(PAUL)             | St. Paul             | the apostle Paul of Tarsus; possible attributes: book, scroll, sword                                      |
| 11H(ANTONY ABBOT)     | St. Anthony Abbot    | the hermit Antony Abbot (Antonius Abbas) of Egypt, also called the Great; possible attributes: bell, book, T-shaped staff, flames, pig |
| 11H(FRANCIS)          | St. Francis of Assisi | founder of the Order of Friars Minor (Franciscans), Francis(cus) of Assisi; possible attributes: book, crucifix, lily, skull, stigmata |
| 11H(JOSEPH)           | St. Joseph           | the foster-father of Christ, Joseph of Nazareth, husband of Mary; possible attributes: flowering rod or wand, lily, carpenter's tools |

## wikidata Results

## wikidata Results
| Model                     | zero-shot (labels)   | few-shot[*](dataset/wikidata-data/few-shot/README.md) (labels)   | zero-shot (descriptions)   |
|:--------------------------|:---------------------|:-----------------------------------------------------------------|:---------------------------|
| clip-vit-base-patch32     | 40.95%               | 40.81%                                                           | 42.20%                     |
| clip-vit-base-patch16     | 48.33%               | 51.11%                                                           | 42.90%                     |
| clip-vit-large-patch14    | 54.46%               | 55.29%                                                           | 50.84%                     |
| siglip-base-patch16-512   | 56.55%               | 56.27%                                                           | 45.40%                     |
| siglip-large-patch16-384  | 60.31%               | 61.84%                                                           | 40.39%                     |
| siglip-so400m-patch14-384 | 65.32%               | 64.35%                                                           | 59.89%                     |
| gpt-4o-mini               | 52.65%               | 46.30%                                                           | 50.00%                     |
| gpt-4o                    | 40.41%               | 75.21%                                                           | 56.96%                     |

# References

[1] Milani, Federico, and Piero Fraternali. "A dataset and a convolutional model for iconography classification in paintings." Journal on Computing and Cultural Heritage (JOCCH) 14.4 (2021): 1-18.

[3] E. Maksimova, M.-A. Meimer, M. Piirsalu, P. Järv. Viability of Zero-shot Classification and Search of Historical Photos CHR2024, Aarhus, Denmark, December 4-6, 2024.

[4] N. Gonthier, «IconArt Dataset». Zenodo, ott. 05, 2018. doi: 10.5281/zenodo.4737435.
