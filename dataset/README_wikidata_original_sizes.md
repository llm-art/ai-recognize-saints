# Wikidata Image Download with Original Sizes

This README explains how to download images from Wikidata while preserving their original sizes and aspect ratios, as opposed to the fixed 512x512 size used in the original implementation.

## Background

The original implementation in `wikidata.ipynb` downloads images and resizes them to a fixed size of 512x512 pixels, which loses the original aspect ratios and dimensions of the paintings. According to the paper you're following, the dataset should have images with dimensions closer to:

- Width: 778.84 ± 198.74 pixels
- Height: 669.36 ± 174.18 pixels

## New Implementation

Two new files have been created to implement image downloading while preserving original sizes and aspect ratios:

1. `wikidata_original_sizes.py` - A standalone Python script
2. `wikidata_original_sizes.ipynb` - A Jupyter notebook with the same functionality

Both implementations:

- Download images while preserving their original dimensions and aspect ratios
- Only resize images if they exceed the maximum pixel limit (to prevent decompression bomb attacks)
- When resizing is necessary, maintain the original aspect ratio
- Track and store image dimensions in the metadata
- Calculate statistics on image dimensions to compare with the target values from the paper
- Create test and ground truth files with the original dimensions included

The Jupyter notebook also includes visualization of the image size distributions.

## How to Use

### Option 1: Run the Python Script

```bash
cd /path/to/LLM-test
python dataset/wikidata_original_sizes.py
```

### Option 2: Run the Jupyter Notebook

Open the `wikidata_original_sizes.ipynb` notebook in Jupyter and run the cells sequentially.

## Output Files

The implementation creates the following output files:

- `wikidata/JPEGImages_original/` - Directory containing the downloaded images with original sizes
- `wikidata/wikidata_original.json` - JSON file with image metadata including dimensions
- `wikidata-data/image_statistics.txt` - Statistics on image dimensions
- `wikidata-data/2_test_original.txt` - List of image filenames
- `wikidata-data/2_ground_truth_original.json` - Ground truth data with image dimensions
- `wikidata-data/image_size_distribution.png` - (Notebook only) Visualization of image size distributions

## Key Differences from Original Implementation

1. **No Forced Resizing**: The original implementation resized all images to 512x512 pixels. The new implementation preserves original dimensions.

2. **Dimension Tracking**: The new implementation tracks and stores the width and height of each image.

3. **Statistics Calculation**: The new implementation calculates statistics on image dimensions to compare with the target values from the paper.

4. **Visualization**: The notebook includes visualization of image size distributions.

5. **Enhanced Metadata**: The ground truth JSON file includes width and height information for each image.

## Notes

- Images that exceed the maximum pixel limit (178,956,970 pixels) will still be resized, but their aspect ratio will be preserved.
- The implementation uses a separate directory (`JPEGImages_original`) to store the images with original sizes, so it won't overwrite the existing images.
