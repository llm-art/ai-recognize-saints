
### IconArt

This dataset contains 5955 images (from WikiCommons): a train set of 2978 images and a test set of 2977 images (for classification task). 

1480 of the 2977 images are annotated with bounding boxes for seven classes: `angel`, `Child_Jesus`, `crucifixion_of_Jesus`, `Mary`, `nudity`, `ruins`, `Saint_Sebastien`. 

Version 2 of the database contains 10 classes with the same images. The classes are `angel`, `beard`, `capital`, `Child_Jesus`, `crucifixion_of_Jesus`, `Mary`, `nudity`, `ruins`, `Saint_Sebastien`, and `turban`.

The dataset is available on [Zenodo](https://zenodo.org/records/4737435). The version used for this experiment is IconArt_v2.zip.

**Note:** Currently, these results are based on a new test set made up of 1955. These images belong to **only one** class.


To handle the test with the description, I found the corresponding ICONCLASS ID and used the related description.

| ID                         | Label                  | Description                                             | ICONCLASS ID       |
|----------------------------|------------------------|---------------------------------------------------------|--------------------|
| Saint_Sebastien           | Saint Sebastien       | The Martyr Sebastian; Possible Attributes: Arrow(s), Bow, Tree-Trunk | 11H(SEBASTIAN)     |
| turban                    | Turban                 | Head-Gear: Turban                                       | 41D221(TURBAN)     |
| crucifixion_of_Jesus      | Crucifixion Of Jesus   | Christ Crucified On A 'Living' Cross                    | 11D356             |
| angel                     | Angel                  | Angels                                                  | 11G                |
| capital                   | Capital                | Capital (~ Column, Pillar)                              | 48C1612            |
| Mary                      | Mary                   | The Virgin Mary                                         | 11F                |
| beard                     | Beard                  | Beard                                                   | 31A534             |
| Child_Jesus               | Child Jesus            | Christ As Child Or Youth (In General) ~ Christian Religion | 11D2               |
| nudity                    | Nudity                 | The (Nude) Human Figure; 'Corpo Humano' (Ripa)         | 31A                |
| ruins                     | Ruins                  | Ruin Of A Building ~ Architecture                      | 48C149             |


**Table X.**

#### Results

| Model                     | zero-shot (labels)   | few-shot[*](dataset/IconArt.ipynb) (labels)   | zero-shot (descriptions)   |
|:--------------------------|:---------------------|:----------------------------------------------|:---------------------------|
| clip-vit-base-patch32     | 7.05%                | 5.79%                                         | 8.49%                      |
| clip-vit-base-patch16     | 7.14%                | 7.72%                                         | 8.30%                      |
| clip-vit-large-patch14    | 6.37%                | 5.21%                                         | 6.76%                      |
| siglip-base-patch16-512   | 7.53%                | 7.53%                                         | 8.30%                      |
| siglip-large-patch16-384  | 7.72%                | 8.01%                                         | 7.82%                      |
| siglip-so400m-patch14-384 | 6.76%                | 6.56%                                         | 8.49%                      |
| gpt-4o-mini               | 9.07%                | -                                             | 9.51%                      |
| gpt-4o                    | 10.91%               | -                                             | 12.26%                     |


**Table X.**

