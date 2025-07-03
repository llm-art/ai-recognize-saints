# ChatGPT Testing for ICONCLASS Dataset

This directory contains prompts and test images for manually verifying [`execute_gemini.py`](../../../script/execute_gemini.py) results using the ChatGPT web interface.

## 📁 Directory Structure

```
prompts/ICONCLASS/chatgpt/
├── README.md                    # This file
├── batch_order.md              # Batch organization and API results summary
├── test_1/
│   └── test_1_prompts.md       # Label-based classification prompts
├── test_2/
│   └── test_2_prompts.md       # Description-based classification prompts
├── test_3/
│   └── test_3_prompts.md       # Few-shot learning classification prompts
├── images/
│   ├── IIHIM_-1548783294.jpg   # Test images (10 total)
│   ├── IIHIM_-1578407314.jpg
│   ├── biblia_sacra_20021227082.jpg
│   ├── IIHIM_-708292484.jpg
│   ├── biblia_sacra_20030130049.jpg
│   ├── IIHIM_-487595164.jpg
│   ├── biblia_sacra_20030110097.jpg
│   ├── IIHIM_-512769350.jpg
│   ├── IIHIM_-1057388368.jpg
│   └── IIHIM_RIJKS_1878098591.jpg
└── images/few-shot/
    ├── 1942_9_17_c.jpg         # Few-shot example images (5 total)
    ├── RKD Research De apostel Paulus een brief schrijvend, jaren 1630.jpg
    ├── RKD Research De Heilige Catharina, eerste helft 16de eeuw.jpg
    ├── RKD Research De heilige Hieronymus.jpg
    └── RKD Research De Heilige Maria Magdalena met zalfpot en boek, eerste helft 16de eeuw.jpg
```

## 🎯 Purpose

Verify the consistency between:
- **API Results**: [`execute_gemini.py`](../../../script/execute_gemini.py) using GPT-4o via OpenAI API
- **Web Interface Results**: ChatGPT web interface using the same prompts and images

## 📊 Test Configuration

### Test Images
- **Source**: ICONCLASS dataset successful API batches (Batch 3 & 4)
- **Total**: 10 images (2 batches of 5 each)
- **Selection Criteria**: Images with successful API classifications
- **Saint Classes**: Catherine (3), Mary Magdalene (3), John (2), Paul (1), Matthew (1)

### Test Types
1. **Test 1**: Label-based classification using short saint names
2. **Test 2**: Description-based classification using detailed saint descriptions  
3. **Test 3**: Few-shot learning using 5 example images

## 🚀 Usage Instructions

### Step 1: Choose Test Type
Navigate to the appropriate test directory:
- [`test_1/`](test_1/) for label-based classification
- [`test_2/`](test_2/) for description-based classification
- [`test_3/`](test_3/) for few-shot learning

### Step 2: Copy Prompt
Open the test's markdown file and copy the prompt section to ChatGPT web interface.

### Step 3: Upload Images
- **Test 1 & 2**: Upload images in 2 batches of 5 each
- **Test 3**: Upload 5 few-shot examples first, then test images in batches

### Step 4: Record Results
Fill in the "ChatGPT Predicted" column in the results tables.

### Step 5: Compare Results
Analyze differences between API and web interface predictions.

## 📈 Expected API Performance

| Test | Batch 1 | Batch 2 | Overall | Notes |
|------|---------|---------|---------|-------|
| **Test 1** (Labels) | 5/5 (100%) | 3/5 (60%) | 8/10 (80%) | Baseline performance |
| **Test 2** (Descriptions) | N/A | 2/5 (40%) | Lower | More errors with descriptions |
| **Test 3** (Few-shot) | 5/5 (100%) | 3/5 (60%) | 8/10 (80%) | Same as Test 1 |

## 🔍 Key Observations

### Consistent Patterns
- **Perfect Batch 1**: All tests achieve 100% accuracy on first 5 images
- **Challenging Batch 2**: Consistent errors on images 8 and 9 across tests
- **Error Types**:
  - Image 8: Catherine → Mary Magdalene (visual similarity)
  - Image 9: Matthew → Peter/Paul (iconographic confusion)

### Test-Specific Insights
- **Labels vs Descriptions**: Simpler labels may perform better than detailed descriptions
- **Few-shot Learning**: Achieves same performance as labels, provides visual context
- **Confidence Scores**: Vary across test types, generally 0.5-0.9 range

## 📋 Analysis Framework

### Metrics to Track
1. **Accuracy**: Correct predictions / Total predictions
2. **Confidence**: Average confidence scores
3. **Error Patterns**: Systematic misclassifications
4. **Consistency**: API vs Web interface agreement

### Comparison Dimensions
- **Accuracy**: Does web interface match API performance?
- **Confidence**: Are confidence scores similar?
- **Error Types**: Do both make the same mistakes?
- **Response Format**: Is JSON formatting consistent?

## 🛠️ Troubleshooting

### Common Issues
- **Image Upload**: Ensure images are uploaded in correct order
- **Prompt Format**: Copy entire prompt including JSON format requirements
- **Batch Size**: Stick to 5 images per batch for consistency
- **File Names**: Use exact filenames as shown in tables

### Validation Checklist
- [ ] Correct prompt copied to ChatGPT
- [ ] Images uploaded in specified order
- [ ] JSON response received
- [ ] Results recorded in markdown tables
- [ ] Confidence scores noted
- [ ] Errors documented and analyzed

## 📚 Related Files

- [`../../../script/execute_gemini.py`](../../../script/execute_gemini.py) - Main classification script
- [`../../../dataset/ICONCLASS-data/`](../../../dataset/ICONCLASS-data/) - Dataset source
- [`../../base_prompt_template.txt`](../../base_prompt_template.txt) - Prompt template
- [`../../../test_1/ICONCLASS/gpt-4o/batches/`](../../../test_1/ICONCLASS/gpt-4o/batches/) - API results

## 📝 Notes

- All images are historical religious artwork used for academic research
- Results should be documented for reproducibility
- Focus on systematic differences rather than individual errors
- Consider both accuracy and confidence in analysis