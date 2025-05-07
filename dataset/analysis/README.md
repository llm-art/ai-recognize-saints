# Cross-Dataset Image Similarity Analysis

## Overview

This analysis examines the similarity between images across different datasets using perceptual and robust hashing techniques. The focus is on identifying similar images between different datasets, rather than within the same dataset.

## Datasets Analyzed

The following datasets were analyzed:

- ArtDL (1864 images)
- ICONCLASS (592 images)
- wikidata (718 images)


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

### Perceptual Hash Results

**Total duplicate images found: 46**

#### Duplicates by Dataset Pair

- **ArtDL - wikidata**: 44 duplicates
  - Examples:
    - 1939_1_291 ↔ Q20173065
    - 1939_1_80 ↔ Q20173671
    - 1950_11_1_a ↔ Q20173413
    - 253141 ↔ Q3947314
    - 253669 ↔ Q20540321
    - 258398 ↔ Q19820268
    - 273854 ↔ Q6004260
    - 408706434 ↔ Q20173671
    - Q15974339 ↔ Q15974339
    - Q17335796 ↔ Q17335796
    - ... and 34 more
- **ICONCLASS - wikidata**: 2 duplicates
  - Examples:
    - IIHIM_838250489 ↔ Q63191747
    - IIHIM_RIJKS_-649904531 ↔ Q17328232



### Robust Hash Results

**Total duplicate images found: 35**

#### Duplicates by Dataset Pair

- **ArtDL - ICONCLASS**: 2 duplicates
  - Examples:
    - ICCD3163621_13815-H ↔ IIHIM_RIJKS_1401436342
    - ICCD3710537_375754 ↔ IIHIM_RIJKS_1827277148
- **ArtDL - wikidata**: 32 duplicates
  - Examples:
    - 1939_1_291 ↔ Q20173065
    - 1939_1_80 ↔ Q20173671
    - 1950_11_1_a ↔ Q20173413
    - 253141 ↔ Q3947314
    - 253669 ↔ Q20540321
    - 258398 ↔ Q19820268
    - Q15974339 ↔ Q15974339
    - Q17335796 ↔ Q17335796
    - Q18748614 ↔ Q18748614
    - Q19925792 ↔ Q19925792
    - ... and 22 more
- **ICONCLASS - wikidata**: 1 duplicates
  - Examples:
    - IIHIM_RIJKS_-649904531 ↔ Q17328232


### Cross-Dataset Duplicates

The following files contain information about cross-dataset similarities:

- **Consolidated duplicates**:
  - Perceptual hash duplicates: `perceptual_cross_duplicates.json`
  - Robust hash duplicates: `robust_cross_duplicates.json`


## Visual Examples of Similar Images

Below are examples of similar images found across different datasets. Each pair shows the images side by side with their dataset and filename information.

<style>
.image-pair {
    display: flex;
    margin-bottom: 20px;
    border: 1px solid #ddd;
    padding: 10px;
    border-radius: 5px;
}
.image-container {
    flex: 1;
    padding: 10px;
    text-align: center;
}
.image-container img {
    max-width: 100%;
    max-height: 300px;
    border: 1px solid #ccc;
}
.image-info {
    margin-top: 10px;
    font-size: 0.9em;
}
</style>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/1939_1_291.jpg" alt="1939_1_291">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> 1939_1_291
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q20173065.jpg" alt="Q20173065">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q20173065
        </div>
    </div>
</div>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/1939_1_80.jpg" alt="1939_1_80">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> 1939_1_80
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q20173671.jpg" alt="Q20173671">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q20173671
        </div>
    </div>
</div>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/1950_11_1_a.jpg" alt="1950_11_1_a">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> 1950_11_1_a
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q20173413.jpg" alt="Q20173413">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q20173413
        </div>
    </div>
</div>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/253141.jpg" alt="253141">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> 253141
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q3947314.jpg" alt="Q3947314">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q3947314
        </div>
    </div>
</div>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/253669.jpg" alt="253669">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> 253669
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q20540321.jpg" alt="Q20540321">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q20540321
        </div>
    </div>
</div>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/258398.jpg" alt="258398">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> 258398
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q19820268.jpg" alt="Q19820268">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q19820268
        </div>
    </div>
</div>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/273854.jpg" alt="273854">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> 273854
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q6004260.jpg" alt="Q6004260">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q6004260
        </div>
    </div>
</div>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/408706434.jpg" alt="408706434">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> 408706434
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q20173671.jpg" alt="Q20173671">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q20173671
        </div>
    </div>
</div>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/Q15974339.jpg" alt="Q15974339">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> Q15974339
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q15974339.jpg" alt="Q15974339">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q15974339
        </div>
    </div>
</div>

<div class="image-pair">
    <div class="image-container">
        <img src="../dataset/ArtDL/JPEGImages/Q17335796.jpg" alt="Q17335796">
        <div class="image-info">
            <strong>Dataset:</strong> ArtDL<br>
            <strong>Filename:</strong> Q17335796
        </div>
    </div>
    <div class="image-container">
        <img src="../dataset/wikidata/JPEGImages/Q17335796.jpg" alt="Q17335796">
        <div class="image-info">
            <strong>Dataset:</strong> wikidata<br>
            <strong>Filename:</strong> Q17335796
        </div>
    </div>
</div>


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
