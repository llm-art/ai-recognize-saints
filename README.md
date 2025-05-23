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


# References

[1] Milani, Federico, and Piero Fraternali. "A dataset and a convolutional model for iconography classification in paintings." Journal on Computing and Cultural Heritage (JOCCH) 14.4 (2021): 1-18.

[3] E. Maksimova, M.-A. Meimer, M. Piirsalu, P. Järv. Viability of Zero-shot Classification and Search of Historical Photos CHR2024, Aarhus, Denmark, December 4-6, 2024.

[4] N. Gonthier, «IconArt Dataset». Zenodo, ott. 05, 2018. doi: 10.5281/zenodo.4737435.
