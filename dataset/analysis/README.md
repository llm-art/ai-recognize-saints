# Cross-Dataset Image Similarity Analysis

## Overview

This analysis examines the similarity between images across different datasets using perceptual and robust hashing techniques. The focus is on identifying similar images between different datasets, rather than within the same dataset.

## Datasets Analyzed

The following datasets were analyzed:

- ArtDL (1864 images)
- ICONCLASS (863 images)
- wikidata (717 images)


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
| ![eyck_van_jan_02page_31jerom](examples/ArtDL_eyck_van_jan_02page_31jerom.jpg) <br> **Dataset:** ArtDL <br> **Filename:** eyck_van_jan_02page_31jerom | ![IIHIM_1441633156](examples/ICONCLASS_IIHIM_1441633156.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_1441633156 |
| ![1939_1_291](examples/ArtDL_1939_1_291.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1939_1_291 | ![Q20173065](examples/wikidata_Q20173065.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173065 |
| ![1950_11_1_a](examples/ArtDL_1950_11_1_a.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1950_11_1_a | ![Q20173413](examples/wikidata_Q20173413.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173413 |
| ![253141](examples/ArtDL_253141.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 253141 | ![Q3947314](examples/wikidata_Q3947314.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q3947314 |
| ![258398](examples/ArtDL_258398.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 258398 | ![Q19820268](examples/wikidata_Q19820268.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19820268 |
| ![273854](examples/ArtDL_273854.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 273854 | ![Q6004260](examples/wikidata_Q6004260.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q6004260 |
| ![Q15974339](examples/ArtDL_Q15974339.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q15974339 | ![Q15974339](examples/wikidata_Q15974339.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q15974339 |
| ![Q17335796](examples/ArtDL_Q17335796.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q17335796 | ![Q17335796](examples/wikidata_Q17335796.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17335796 |
| ![Q18225338](examples/ArtDL_Q18225338.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q18225338 | ![Q18225338](examples/wikidata_Q18225338.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q18225338 |
| ![Q18748614](examples/ArtDL_Q18748614.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q18748614 | ![Q18748614](examples/wikidata_Q18748614.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q18748614 |
| ![Q19926040](examples/ArtDL_Q19926040.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q19926040 | ![Q19926040](examples/wikidata_Q19926040.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19926040 |
| ![Q20173413](examples/ArtDL_Q20173413.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20173413 | ![Q20173413](examples/wikidata_Q20173413.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173413 |
| ![Q20173883](examples/ArtDL_Q20173883.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20173883 | ![Q20173883](examples/wikidata_Q20173883.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173883 |
| ![Q20267955](examples/ArtDL_Q20267955.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20267955 | ![Q20267955](examples/wikidata_Q20267955.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20267955 |
| ![Q21283213](examples/ArtDL_Q21283213.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q21283213 | ![Q21283213](examples/wikidata_Q21283213.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q21283213 |
| ![Q2715177](examples/ArtDL_Q2715177.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q2715177 | ![Q2715177](examples/wikidata_Q2715177.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q2715177 |
| ![Q27981491](examples/ArtDL_Q27981491.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q27981491 | ![Q27981491](examples/wikidata_Q27981491.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q27981491 |
| ![Q29024815](examples/ArtDL_Q29024815.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q29024815 | ![Q29024815](examples/wikidata_Q29024815.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29024815 |
| ![Q29477236](examples/ArtDL_Q29477236.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q29477236 | ![Q29477236](examples/wikidata_Q29477236.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29477236 |
| ![Q4448822](examples/ArtDL_Q4448822.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q4448822 | ![Q4448822](examples/wikidata_Q4448822.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q4448822 |
| ![Q4448822](examples/ArtDL_Q4448822.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q4448822 | ![Q17321337](examples/wikidata_Q17321337.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17321337 |
| ![Q4448822](examples/ArtDL_Q4448822.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q4448822 | ![Q29939692](examples/wikidata_Q29939692.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29939692 |
| ![Q510799](examples/ArtDL_Q510799.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q510799 | ![Q510799](examples/wikidata_Q510799.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q510799 |
| ![Q55102676](examples/ArtDL_Q55102676.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q55102676 | ![Q55102676](examples/wikidata_Q55102676.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q55102676 |
| ![Q6004260](examples/ArtDL_Q6004260.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q6004260 | ![Q6004260](examples/wikidata_Q6004260.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q6004260 |
| ![Q9073676](examples/ArtDL_Q9073676.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q9073676 | ![Q9073676](examples/wikidata_Q9073676.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q9073676 |
| ![_C_438722921_7632](examples/ArtDL__C_438722921_7632.jpg) <br> **Dataset:** ArtDL <br> **Filename:** _C_438722921_7632 | ![Q9015206](examples/wikidata_Q9015206.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q9015206 |
| ![__EX_1000788252_18423](examples/ArtDL___EX_1000788252_18423.jpg) <br> **Dataset:** ArtDL <br> **Filename:** __EX_1000788252_18423 | ![Q20172983](examples/wikidata_Q20172983.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20172983 |
| ![caravagg_07_44jerome](examples/ArtDL_caravagg_07_44jerome.jpg) <br> **Dataset:** ArtDL <br> **Filename:** caravagg_07_44jerome | ![Q2715177](examples/wikidata_Q2715177.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q2715177 |
| ![caravagg_11_71baptis](examples/ArtDL_caravagg_11_71baptis.jpg) <br> **Dataset:** ArtDL <br> **Filename:** caravagg_11_71baptis | ![Q9015206](examples/wikidata_Q9015206.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q9015206 |
| ![clouet_jean_francbap](examples/ArtDL_clouet_jean_francbap.jpg) <br> **Dataset:** ArtDL <br> **Filename:** clouet_jean_francbap | ![Q30096142](examples/wikidata_Q30096142.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q30096142 |
| ![en-SK-A-3382](examples/ArtDL_en-SK-A-3382.jpg) <br> **Dataset:** ArtDL <br> **Filename:** en-SK-A-3382 | ![Q17334273](examples/wikidata_Q17334273.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17334273 |
| ![en-SK-A-4006](examples/ArtDL_en-SK-A-4006.jpg) <br> **Dataset:** ArtDL <br> **Filename:** en-SK-A-4006 | ![Q17335839](examples/wikidata_Q17335839.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17335839 |
| ![greco_el_06_0603grec](examples/ArtDL_greco_el_06_0603grec.jpg) <br> **Dataset:** ArtDL <br> **Filename:** greco_el_06_0603grec | ![Q9026835](examples/wikidata_Q9026835.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q9026835 |
| ![greco_el_17_1703grec](examples/ArtDL_greco_el_17_1703grec.jpg) <br> **Dataset:** ArtDL <br> **Filename:** greco_el_17_1703grec | ![Q16589363](examples/wikidata_Q16589363.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q16589363 |
| ![hemessen_jan_stjerom](examples/ArtDL_hemessen_jan_stjerom.jpg) <br> **Dataset:** ArtDL <br> **Filename:** hemessen_jan_stjerom | ![Q114744953](examples/wikidata_Q114744953.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q114744953 |
| ![la_tour_georges_1_10penite](examples/ArtDL_la_tour_georges_1_10penite.jpg) <br> **Dataset:** ArtDL <br> **Filename:** la_tour_georges_1_10penite | ![Q3210251](examples/wikidata_Q3210251.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q3210251 |
| ![nogari_apostle](examples/ArtDL_nogari_apostle.jpg) <br> **Dataset:** ArtDL <br> **Filename:** nogari_apostle | ![Q50326658](examples/wikidata_Q50326658.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q50326658 |
| ![piero_co_z_other_mary_mag](examples/ArtDL_piero_co_z_other_mary_mag.jpg) <br> **Dataset:** ArtDL <br> **Filename:** piero_co_z_other_mary_mag | ![Q28229479](examples/wikidata_Q28229479.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q28229479 |
| ![tintoret_3b_3ground_5maryma](examples/ArtDL_tintoret_3b_3ground_5maryma.jpg) <br> **Dataset:** ArtDL <br> **Filename:** tintoret_3b_3ground_5maryma | ![Q11769022](examples/wikidata_Q11769022.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q11769022 |
| ![IIHIM_-1578407314](examples/ICONCLASS_IIHIM_-1578407314.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_-1578407314 | ![Q107443479](examples/wikidata_Q107443479.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q107443479 |
| ![IIHIM_1359909329](examples/ICONCLASS_IIHIM_1359909329.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_1359909329 | ![Q117226027](examples/wikidata_Q117226027.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q117226027 |
| ![IIHIM_838250489](examples/ICONCLASS_IIHIM_838250489.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_838250489 | ![Q63191747](examples/wikidata_Q63191747.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q63191747 |
| ![IIHIM_-1583071816](examples/ICONCLASS_IIHIM_-1583071816.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_-1583071816 | ![Q29648941](examples/wikidata_Q29648941.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29648941 |
| ![IIHIM_RIJKS_2033920572](examples/ICONCLASS_IIHIM_RIJKS_2033920572.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_RIJKS_2033920572 | ![Q17347293](examples/wikidata_Q17347293.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17347293 |

### Robust Hash Pairs

| Image 1 | Image 2 |
|---------|---------|
| ![ICCD3163621_13815-H](examples/ArtDL_ICCD3163621_13815-H.jpg) <br> **Dataset:** ArtDL <br> **Filename:** ICCD3163621_13815-H | ![IIHIM_-1335425534](examples/ICONCLASS_IIHIM_-1335425534.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_-1335425534 |
| ![ICCD3163621_13815-H](examples/ArtDL_ICCD3163621_13815-H.jpg) <br> **Dataset:** ArtDL <br> **Filename:** ICCD3163621_13815-H | ![IIHIM_RIJKS_1401436342](examples/ICONCLASS_IIHIM_RIJKS_1401436342.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_RIJKS_1401436342 |
| ![ICCD3710537_375754](examples/ArtDL_ICCD3710537_375754.jpg) <br> **Dataset:** ArtDL <br> **Filename:** ICCD3710537_375754 | ![IIHIM_RIJKS_1827277148](examples/ICONCLASS_IIHIM_RIJKS_1827277148.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_RIJKS_1827277148 |
| ![ICCD4203971_00069043](examples/ArtDL_ICCD4203971_00069043.jpg) <br> **Dataset:** ArtDL <br> **Filename:** ICCD4203971_00069043 | ![IIHIM_-1335425534](examples/ICONCLASS_IIHIM_-1335425534.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_-1335425534 |
| ![1939_1_291](examples/ArtDL_1939_1_291.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1939_1_291 | ![Q20173065](examples/wikidata_Q20173065.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173065 |
| ![1950_11_1_a](examples/ArtDL_1950_11_1_a.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 1950_11_1_a | ![Q20173413](examples/wikidata_Q20173413.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173413 |
| ![253141](examples/ArtDL_253141.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 253141 | ![Q3947314](examples/wikidata_Q3947314.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q3947314 |
| ![258398](examples/ArtDL_258398.jpg) <br> **Dataset:** ArtDL <br> **Filename:** 258398 | ![Q19820268](examples/wikidata_Q19820268.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19820268 |
| ![Q15974339](examples/ArtDL_Q15974339.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q15974339 | ![Q15974339](examples/wikidata_Q15974339.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q15974339 |
| ![Q17335796](examples/ArtDL_Q17335796.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q17335796 | ![Q17335796](examples/wikidata_Q17335796.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17335796 |
| ![Q18748614](examples/ArtDL_Q18748614.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q18748614 | ![Q18748614](examples/wikidata_Q18748614.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q18748614 |
| ![Q19926040](examples/ArtDL_Q19926040.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q19926040 | ![Q19926040](examples/wikidata_Q19926040.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q19926040 |
| ![Q20173413](examples/ArtDL_Q20173413.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20173413 | ![Q20173413](examples/wikidata_Q20173413.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173413 |
| ![Q20173883](examples/ArtDL_Q20173883.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20173883 | ![Q20173883](examples/wikidata_Q20173883.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20173883 |
| ![Q20267955](examples/ArtDL_Q20267955.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q20267955 | ![Q20267955](examples/wikidata_Q20267955.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20267955 |
| ![Q21283213](examples/ArtDL_Q21283213.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q21283213 | ![Q21283213](examples/wikidata_Q21283213.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q21283213 |
| ![Q2715177](examples/ArtDL_Q2715177.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q2715177 | ![Q2715177](examples/wikidata_Q2715177.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q2715177 |
| ![Q29024815](examples/ArtDL_Q29024815.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q29024815 | ![Q29024815](examples/wikidata_Q29024815.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29024815 |
| ![Q29477236](examples/ArtDL_Q29477236.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q29477236 | ![Q29477236](examples/wikidata_Q29477236.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q29477236 |
| ![Q4448822](examples/ArtDL_Q4448822.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q4448822 | ![Q4448822](examples/wikidata_Q4448822.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q4448822 |
| ![Q510799](examples/ArtDL_Q510799.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q510799 | ![Q510799](examples/wikidata_Q510799.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q510799 |
| ![Q55102676](examples/ArtDL_Q55102676.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q55102676 | ![Q55102676](examples/wikidata_Q55102676.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q55102676 |
| ![Q6004260](examples/ArtDL_Q6004260.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q6004260 | ![Q6004260](examples/wikidata_Q6004260.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q6004260 |
| ![Q9073676](examples/ArtDL_Q9073676.jpg) <br> **Dataset:** ArtDL <br> **Filename:** Q9073676 | ![Q9073676](examples/wikidata_Q9073676.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q9073676 |
| ![__EX_1000788252_18423](examples/ArtDL___EX_1000788252_18423.jpg) <br> **Dataset:** ArtDL <br> **Filename:** __EX_1000788252_18423 | ![Q20172983](examples/wikidata_Q20172983.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q20172983 |
| ![clouet_jean_francbap](examples/ArtDL_clouet_jean_francbap.jpg) <br> **Dataset:** ArtDL <br> **Filename:** clouet_jean_francbap | ![Q30096142](examples/wikidata_Q30096142.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q30096142 |
| ![en-SK-A-3382](examples/ArtDL_en-SK-A-3382.jpg) <br> **Dataset:** ArtDL <br> **Filename:** en-SK-A-3382 | ![Q17334273](examples/wikidata_Q17334273.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17334273 |
| ![en-SK-A-4006](examples/ArtDL_en-SK-A-4006.jpg) <br> **Dataset:** ArtDL <br> **Filename:** en-SK-A-4006 | ![Q17335839](examples/wikidata_Q17335839.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17335839 |
| ![greco_el_17_1703grec](examples/ArtDL_greco_el_17_1703grec.jpg) <br> **Dataset:** ArtDL <br> **Filename:** greco_el_17_1703grec | ![Q16589363](examples/wikidata_Q16589363.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q16589363 |
| ![hemessen_jan_stjerom](examples/ArtDL_hemessen_jan_stjerom.jpg) <br> **Dataset:** ArtDL <br> **Filename:** hemessen_jan_stjerom | ![Q114744953](examples/wikidata_Q114744953.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q114744953 |
| ![IIHIM_1359909329](examples/ICONCLASS_IIHIM_1359909329.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_1359909329 | ![Q117226027](examples/wikidata_Q117226027.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q117226027 |
| ![IIHIM_RIJKS_2033920572](examples/ICONCLASS_IIHIM_RIJKS_2033920572.jpg) <br> **Dataset:** ICONCLASS <br> **Filename:** IIHIM_RIJKS_2033920572 | ![Q17347293](examples/wikidata_Q17347293.jpg) <br> **Dataset:** wikidata <br> **Filename:** Q17347293 |


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
