# Cross-Dataset Image Similarity Analysis

## Overview

This analysis examines the similarity between images across different datasets using perceptual and robust hashing techniques. The focus is on identifying similar images between different datasets, rather than within the same dataset.

## Datasets Analyzed

The following datasets were analyzed:

- ArtDL (1864 images)
- ICONCLASS (863 images)
- wikidata (711 images)


## Methodology

Two different hashing techniques were used to compute image similarity:

1. **Perceptual Hashing (phash, size=8x8)**: 
   - Detects visually similar images based on their appearance
   - Threshold for similarity: 8

2. **Robust Hashing (block mean hash, size=16x16)**: 
   - More robust to minor image transformations
   - Threshold for similarity: 8

## Results

### Dataset Overlap

The Venn diagrams show the overlap between the datasets:

- [Perceptual Hash Similarity Venn Diagram](./perceptual_venn_diagram.png)
- [Robust Hash Similarity Venn Diagram](./robust_venn_diagram.png)

The size of each circle is proportional to the number of images in the dataset, and the overlapping regions show the number of similar images between datasets.

### Cross-Dataset Duplicates

The following files contain information about cross-dataset similarities:

- **Consolidated duplicates**:
  - Perceptual hash duplicates: `perceptual_cross_duplicates.json`
  - Robust hash duplicates: `robust_cross_duplicates.json`


## All Similar Image Pairs

Below are all pairs of similar images found across different datasets.

### Perceptual Hash Pairs

| Image 1 | Image 2 |
|---------|---------|
| ![eyck_van_jan_02page_31jerom](examples/eyck_van_jan_02page_31jerom.jpg) <br> **Dataset:** ArtDL <br> **Filename:** eyck_van_jan_02page_31jerom | ![IIHIM_1441633156](examples/IIHIM_1441633156.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_1441633156 |
| ![1939_1_291](examples/1939_1_291.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1939_1_291 | ![Q20173065](examples/Q20173065.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173065 |
| ![1939_1_80](examples/1939_1_80.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1939_1_80 | ![Q20173671](examples/Q20173671.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173671 |
| ![1950_11_1_a](examples/1950_11_1_a.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1950_11_1_a | ![Q20173413](examples/Q20173413.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173413 |
| ![253141](examples/253141.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 253141 | ![Q3947314](examples/Q3947314.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q3947314 |
| ![253669](examples/253669.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 253669 | ![Q20540321](examples/Q20540321.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20540321 |
| ![258398](examples/258398.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 258398 | ![Q19820268](examples/Q19820268.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19820268 |
| ![273854](examples/273854.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 273854 | ![Q6004260](examples/Q6004260.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q6004260 |
| ![408706434](examples/408706434.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 408706434 | ![Q20173671](examples/Q20173671.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173671 |
| ![Q15974339](examples/Q15974339.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q15974339 | ![Q15974339](examples/Q15974339.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q15974339 |
| ![Q17335796](examples/Q17335796.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q17335796 | ![Q17335796](examples/Q17335796.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17335796 |
| ![Q18225338](examples/Q18225338.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q18225338 | ![Q18225338](examples/Q18225338.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q18225338 |
| ![Q18748614](examples/Q18748614.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q18748614 | ![Q18748614](examples/Q18748614.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q18748614 |
| ![Q19925792](examples/Q19925792.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q19925792 | ![Q19925792](examples/Q19925792.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19925792 |
| ![Q19926040](examples/Q19926040.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q19926040 | ![Q19926040](examples/Q19926040.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19926040 |
| ![Q20173413](examples/ArtDL_Q20173413.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20173413 | ![Q20173413](examples/Q20173413.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173413 |
| ![Q20173883](examples/Q20173883.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20173883 | ![Q20173883](examples/Q20173883.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173883 |
| ![Q20267955](examples/Q20267955.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20267955 | ![Q20267955](examples/Q20267955.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20267955 |
| ![Q21283213](examples/Q21283213.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q21283213 | ![Q21283213](examples/Q21283213.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q21283213 |
| ![Q2715177](examples/Q2715177.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q2715177 | ![Q2715177](examples/Q2715177.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q2715177 |
| ![Q27981491](examples/Q27981491.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q27981491 | ![Q27981491](examples/Q27981491.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q27981491 |
| ![Q29024815](examples/Q29024815.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q29024815 | ![Q29024815](examples/Q29024815.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29024815 |
| ![Q29477236](examples/Q29477236.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q29477236 | ![Q29477236](examples/Q29477236.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29477236 |
| ![Q4448822](examples/Q4448822.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q4448822 | ![Q4448822](examples/Q4448822.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q4448822 |
| ![Q4448822](examples/ArtDL_Q4448822.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q4448822 | ![Q17321337](examples/Q17321337.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17321337 |
| ![Q4448822](examples/ArtDL_Q4448822.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q4448822 | ![Q29939692](examples/Q29939692.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29939692 |
| ![Q510799](examples/Q510799.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q510799 | ![Q510799](examples/Q510799.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q510799 |
| ![Q55102676](examples/Q55102676.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q55102676 | ![Q55102676](examples/Q55102676.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q55102676 |
| ![Q6004260](examples/ArtDL_Q6004260.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q6004260 | ![Q6004260](examples/Q6004260.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q6004260 |
| ![__EX_1000788252_18423](examples/__EX_1000788252_18423.jpg) <br> **Dataset:** ArtDL <br> **Filename:** __EX_1000788252_18423 | ![Q20172983](examples/Q20172983.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20172983 |
| ![caravagg_07_44jerome](examples/caravagg_07_44jerome.jpg) <br> **Dataset:** ArtDL <br> **Filename:** caravagg_07_44jerome | ![Q2715177](examples/Q2715177.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q2715177 |
| ![clouet_jean_francbap](examples/clouet_jean_francbap.jpg) <br> **Dataset:** ArtDL <br> **Filename:** clouet_jean_francbap | ![Q30096142](examples/Q30096142.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q30096142 |
| ![en-SK-A-3382](examples/en-SK-A-3382.jpg) <br> **Dataset:** ArtDL <br> **Filename:** en-SK-A-3382 | ![Q17334273](examples/Q17334273.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17334273 |
| ![en-SK-A-4006](examples/en-SK-A-4006.jpg) <br> **Dataset:** ArtDL <br> **Filename:** en-SK-A-4006 | ![Q17335839](examples/Q17335839.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17335839 |
| ![greco_el_17_1703grec](examples/greco_el_17_1703grec.jpg) <br> **Dataset:** ArtDL <br> **Filename:** greco_el_17_1703grec | ![Q16589363](examples/Q16589363.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q16589363 |
| ![hemessen_jan_stjerom](examples/hemessen_jan_stjerom.jpg) <br> **Dataset:** ArtDL <br> **Filename:** hemessen_jan_stjerom | ![Q114744953](examples/Q114744953.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q114744953 |
| ![la_tour_georges_1_10penite](examples/la_tour_georges_1_10penite.jpg) <br> **Dataset:** ArtDL <br> **Filename:** la_tour_georges_1_10penite | ![Q3210251](examples/Q3210251.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q3210251 |
| ![nogari_apostle](examples/nogari_apostle.jpg) <br> **Dataset:** ArtDL <br> **Filename:** nogari_apostle | ![Q50326658](examples/Q50326658.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q50326658 |
| ![piero_co_z_other_mary_mag](examples/piero_co_z_other_mary_mag.jpg) <br> **Dataset:** ArtDL <br> **Filename:** piero_co_z_other_mary_mag | ![Q28229479](examples/Q28229479.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q28229479 |
| ![tintoret_3b_3ground_5maryma](examples/tintoret_3b_3ground_5maryma.jpg) <br> **Dataset:** ArtDL <br> **Filename:** tintoret_3b_3ground_5maryma | ![Q11769022](examples/Q11769022.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q11769022 |
| ![IIHIM_-1578407314](examples/IIHIM_-1578407314.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_-1578407314 | ![Q107443479](examples/Q107443479.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q107443479 |
| ![IIHIM_1359909329](examples/IIHIM_1359909329.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_1359909329 | ![Q117226027](examples/Q117226027.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q117226027 |
| ![IIHIM_838250489](examples/IIHIM_838250489.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_838250489 | ![Q63191747](examples/Q63191747.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q63191747 |
| ![IIHIM_-1583071816](examples/IIHIM_-1583071816.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_-1583071816 | ![Q29648941](examples/Q29648941.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29648941 |
| ![IIHIM_RIJKS_2033920572](examples/IIHIM_RIJKS_2033920572.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_RIJKS_2033920572 | ![Q17347293](examples/Q17347293.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17347293 |

### Robust Hash Pairs

| Image 1 | Image 2 |
|---------|---------|
| ![ICCD3163621_13815-H](examples/ICCD3163621_13815-H.jpg) <br> **Dataset:** ArtDL <br> **Filename:** ICCD3163621_13815-H | ![IIHIM_-1335425534](examples/IIHIM_-1335425534.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_-1335425534 |
| ![ICCD3163621_13815-H](examples/ICCD3163621_13815-H.jpg) <br> **Dataset:** ArtDL <br> **Filename:** ICCD3163621_13815-H | ![IIHIM_RIJKS_1401436342](examples/IIHIM_RIJKS_1401436342.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_RIJKS_1401436342 |
| ![ICCD3710537_375754](examples/ICCD3710537_375754.jpg) <br> **Dataset:** ArtDL <br> **Filename:** ICCD3710537_375754 | ![IIHIM_RIJKS_1827277148](examples/IIHIM_RIJKS_1827277148.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_RIJKS_1827277148 |
| ![ICCD4203971_00069043](examples/ICCD4203971_00069043.jpg) <br> **Dataset:** ArtDL <br> **Filename:** ICCD4203971_00069043 | ![IIHIM_-1335425534](examples/IIHIM_-1335425534.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_-1335425534 |
| ![1939_1_291](examples/1939_1_291.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1939_1_291 | ![Q20173065](examples/Q20173065.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173065 |
| ![1939_1_80](examples/1939_1_80.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1939_1_80 | ![Q20173671](examples/Q20173671.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173671 |
| ![1950_11_1_a](examples/1950_11_1_a.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1950_11_1_a | ![Q20173413](examples/Q20173413.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173413 |
| ![253141](examples/253141.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 253141 | ![Q3947314](examples/Q3947314.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q3947314 |
| ![253669](examples/253669.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 253669 | ![Q20540321](examples/Q20540321.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20540321 |
| ![258398](examples/258398.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 258398 | ![Q19820268](examples/Q19820268.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19820268 |
| ![Q15974339](examples/ArtDL_Q15974339.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q15974339 | ![Q15974339](examples/Q15974339.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q15974339 |
| ![Q17335796](examples/ArtDL_Q17335796.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q17335796 | ![Q17335796](examples/Q17335796.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17335796 |
| ![Q18748614](examples/ArtDL_Q18748614.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q18748614 | ![Q18748614](examples/Q18748614.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q18748614 |
| ![Q19925792](examples/ArtDL_Q19925792.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q19925792 | ![Q19925792](examples/Q19925792.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19925792 |
| ![Q19926040](examples/ArtDL_Q19926040.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q19926040 | ![Q19926040](examples/Q19926040.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19926040 |
| ![Q20173413](examples/ArtDL_Q20173413.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20173413 | ![Q20173413](examples/Q20173413.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173413 |
| ![Q20173883](examples/ArtDL_Q20173883.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20173883 | ![Q20173883](examples/Q20173883.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173883 |
| ![Q20267955](examples/ArtDL_Q20267955.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20267955 | ![Q20267955](examples/Q20267955.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20267955 |
| ![Q21283213](examples/ArtDL_Q21283213.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q21283213 | ![Q21283213](examples/Q21283213.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q21283213 |
| ![Q2715177](examples/ArtDL_Q2715177.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q2715177 | ![Q2715177](examples/Q2715177.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q2715177 |
| ![Q27981491](examples/ArtDL_Q27981491.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q27981491 | ![Q27981491](examples/Q27981491.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q27981491 |
| ![Q29024815](examples/ArtDL_Q29024815.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q29024815 | ![Q29024815](examples/Q29024815.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29024815 |
| ![Q29477236](examples/ArtDL_Q29477236.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q29477236 | ![Q29477236](examples/Q29477236.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29477236 |
| ![Q4448822](examples/ArtDL_Q4448822.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q4448822 | ![Q4448822](examples/Q4448822.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q4448822 |
| ![Q510799](examples/ArtDL_Q510799.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q510799 | ![Q510799](examples/Q510799.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q510799 |
| ![Q55102676](examples/ArtDL_Q55102676.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q55102676 | ![Q55102676](examples/Q55102676.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q55102676 |
| ![Q6004260](examples/ArtDL_Q6004260.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q6004260 | ![Q6004260](examples/Q6004260.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q6004260 |
| ![__EX_1000788252_18423](examples/__EX_1000788252_18423.jpg) <br> **Dataset:** ArtDL <br> **Filename:** __EX_1000788252_18423 | ![Q20172983](examples/Q20172983.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20172983 |
| ![clouet_jean_francbap](examples/clouet_jean_francbap.jpg) <br> **Dataset:** ArtDL <br> **Filename:** clouet_jean_francbap | ![Q30096142](examples/Q30096142.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q30096142 |
| ![en-SK-A-3382](examples/en-SK-A-3382.jpg) <br> **Dataset:** ArtDL <br> **Filename:** en-SK-A-3382 | ![Q17334273](examples/Q17334273.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17334273 |
| ![en-SK-A-4006](examples/en-SK-A-4006.jpg) <br> **Dataset:** ArtDL <br> **Filename:** en-SK-A-4006 | ![Q17335839](examples/Q17335839.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17335839 |
| ![greco_el_17_1703grec](examples/greco_el_17_1703grec.jpg) <br> **Dataset:** ArtDL <br> **Filename:** greco_el_17_1703grec | ![Q16589363](examples/Q16589363.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q16589363 |
| ![hemessen_jan_stjerom](examples/hemessen_jan_stjerom.jpg) <br> **Dataset:** ArtDL <br> **Filename:** hemessen_jan_stjerom | ![Q114744953](examples/Q114744953.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q114744953 |
| ![tintoret_3b_3ground_5maryma](examples/tintoret_3b_3ground_5maryma.jpg) <br> **Dataset:** ArtDL <br> **Filename:** tintoret_3b_3ground_5maryma | ![Q11769022](examples/Q11769022.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q11769022 |
| ![IIHIM_1359909329](examples/IIHIM_1359909329.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_1359909329 | ![Q117226027](examples/Q117226027.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q117226027 |
| ![IIHIM_RIJKS_2033920572](examples/IIHIM_RIJKS_2033920572.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_RIJKS_2033920572 | ![Q17347293](examples/Q17347293.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17347293 |


## Dataset-specific Files

Each dataset has its own directory with the following files:

- **ArtDL**:
  - Perceptual hashes: `ArtDL/perceptual_hashes.json`
  - Robust hashes: `ArtDL/robust_hashes.json`
- **ICONCLASS**:
  - Perceptual hashes: `ICONCLASS/perceptual_hashes.json`
  - Robust hashes: `ICONCLASS/robust_hashes.json`
- **wikidata**:
  - Perceptual hashes: `wikidata/perceptual_hashes.json`
  - Robust hashes: `wikidata/robust_hashes.json`


## Summary

This analysis focused on cross-dataset image similarity, computing hashes for all images and comparing them across different datasets. The results provide insights into the overlap between datasets and can be used to identify duplicate or similar images across collections.
