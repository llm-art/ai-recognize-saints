# Wikidata Image Download with Original Sizes (Updated)

This README explains the updated implementation for downloading images from Wikidata while preserving their original sizes and aspect ratios.

## Updates in this Version

The updated implementation addresses several issues and adds new features:

1. **Fixed PIL DecompressionBombWarning**: Increased the PIL threshold to match our max_pixels value (178,956,970 pixels) to prevent warnings.

2. **Changed Output Directory**: Images are now saved to `/home/ubuntu/gspinaci/LLM-test/dataset/wikidata/JPEGImages` as requested.

3. **Added Verification**: Added verification after downloading to ensure images are saved successfully.

4. **Tracked Failed Downloads**: Failed downloads are now tracked and saved to a separate JSON file for analysis.

5. **Enhanced Logging**: Added detailed logging to help diagnose issues.

6. **Added Analysis of Failed Downloads**: The notebook now includes a section to analyze failed downloads.

## Files

1. `wikidata_original_sizes_updated.py` - Updated Python script
2. `wikidata_original_sizes_updated.ipynb` - Updated Jupyter notebook

## How to Use

### Option 1: Run the Python Script

```bash
cd /path/to/LLM-test
python dataset/wikidata_original_sizes_updated.py
```

### Option 2: Run the Jupyter Notebook

Open the `wikidata_original_sizes_updated.ipynb` notebook in Jupyter and run the cells sequentially.

## Output Files

The implementation creates the following output files:

- `/home/ubuntu/gspinaci/LLM-test/dataset/wikidata/JPEGImages/` - Directory containing the downloaded images with original sizes
- `wikidata/wikidata_original.json` - JSON file with image metadata including dimensions
- `wikidata/failed_downloads.json` - JSON file with information about failed downloads
- `wikidata-data/image_statistics.txt` - Statistics on image dimensions
- `wikidata-data/2_test_original.txt` - List of image filenames
- `wikidata-data/2_ground_truth_original.json` - Ground truth data with image dimensions
- `wikidata-data/image_size_distribution.png` - (Notebook only) Visualization of image size distributions
- `image_download.log` - Detailed log of the download process

## Key Features

1. **Preserved Original Dimensions**: Images are downloaded with their original dimensions and aspect ratios.

2. **Safe Resizing**: Images that exceed the maximum pixel limit (178,956,970 pixels) are resized while maintaining their aspect ratio.

3. **Increased PIL Threshold**: The PIL threshold is increased to match our max_pixels value to prevent warnings.

4. **Verification**: Each image is verified after downloading to ensure it was saved successfully.

5. **Failed Download Tracking**: Failed downloads are tracked and saved to a separate JSON file for analysis.

6. **Enhanced Logging**: Detailed logging is added to help diagnose issues.

7. **Statistics and Visualization**: The implementation calculates statistics on image dimensions and provides visualizations to compare with the target values from the paper.

## Handling Large Images

The implementation handles large images by:

1. Increasing the PIL threshold to match our max_pixels value (178,956,970 pixels)
2. Resizing images that exceed the maximum pixel limit while maintaining their aspect ratio
3. Verifying that images are saved successfully

## Analyzing Failed Downloads

The notebook includes a section to analyze failed downloads, which can help identify patterns or issues with specific images or iconclasses.
