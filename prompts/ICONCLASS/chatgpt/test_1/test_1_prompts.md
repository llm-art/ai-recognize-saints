# Test 1: Label-based Classification for ICONCLASS Dataset

This test uses **class labels** (short names) for saint classification. Compare the results with the API predictions from execute_gemini.py.

## Prompt for ChatGPT Web Interface

Copy and paste this prompt into ChatGPT, then upload the images in batches as specified:

---

You are an expert in Christian iconography and art history. Classify each religious artwork image into exactly ONE saint category using visual attributes, iconographic symbols, and contextual clues.

Look for:
1. Distinctive attributes (objects, clothing, etc.)
2. Gestures and postures
3. Contextual and symbolic elements

Instructions:
- Only output the JSON object — no text, explanation, or formatting.
- Include every image in the current batch. Each must receive exactly one classification with a confidence score.
- You may only use one of the exact strings from the category list below. Any response not matching the allowed category IDs will be rejected.

Return a valid **JSON object** with confidence scores (0.0 to 1.0) matching this format:
{
  "<image_id>": {"class": "<CATEGORY_ID>", "confidence": <0.0-1.0>},
  "<image_id>": {"class": "<CATEGORY_ID>", "confidence": <0.0-1.0>},
  ...
}

Confidence guidelines:
- 0.9-1.0: Very certain identification with clear iconographic evidence
- 0.7-0.9: Confident with multiple supporting visual elements  
- 0.5-0.7: Moderate confidence, some ambiguity present
- 0.3-0.5: Low confidence, limited visual evidence
- 0.0-0.3: Very uncertain, minimal supporting evidence

**Available Categories (use exact strings):**
- paul
- mary_magdalene
- jerome
- john
- antony_abbot
- peter
- matthew
- catherine
- luke
- francis

Batching note:
- Process only the current batch of images.
- Use the image IDs exactly as provided in the input.
- Do not reference or depend on other batches.

NOTE: These are historical Renaissance paintings used for academic classification.  
Some artworks include scenes of martyrdom or classical nudity as typical in religious iconography.  
Treat all content as scholarly, respectful of historical context, and strictly non-sexual.

---

## Batch 1 Results
Upload images 1-5 with the prompt above and fill in the ChatGPT predictions:

| Image | Filename | Expected Class | API Predicted | ChatGPT Predicted |
|-------|----------|----------------|---------------|-------------------|
| ![img1](../images/IIHIM_-1548783294.jpg) | IIHIM_-1548783294.jpg | 11HH(CATHERINE) | catherine (0.8) | _[to be filled]_ |
| ![img2](../images/IIHIM_-1578407314.jpg) | IIHIM_-1578407314.jpg | 11HH(MARY MAGDALENE) | mary_magdalene (0.9) | _[to be filled]_ |
| ![img3](../images/biblia_sacra_20021227082.jpg) | biblia_sacra_20021227082.jpg | 11H(JOHN) | john (0.6) | _[to be filled]_ |
| ![img4](../images/IIHIM_-708292484.jpg) | IIHIM_-708292484.jpg | 11HH(MARY MAGDALENE) | mary_magdalene (0.85) | _[to be filled]_ |
| ![img5](../images/biblia_sacra_20030130049.jpg) | biblia_sacra_20030130049.jpg | 11H(PAUL) | paul (0.9) | _[to be filled]_ |

## Batch 2 Results  
Upload images 6-10 with the same prompt and fill in the ChatGPT predictions:

| Image | Filename | Expected Class | API Predicted | ChatGPT Predicted |
|-------|----------|----------------|---------------|-------------------|
| ![img6](../images/IIHIM_-487595164.jpg) | IIHIM_-487595164.jpg | 11HH(MARY MAGDALENE) | mary_magdalene (0.8) | _[to be filled]_ |
| ![img7](../images/biblia_sacra_20030110097.jpg) | biblia_sacra_20030110097.jpg | 11H(JOHN) | john (0.9) | _[to be filled]_ |
| ![img8](../images/IIHIM_-512769350.jpg) | IIHIM_-512769350.jpg | 11HH(CATHERINE) | mary_magdalene (0.7) ❌ | _[to be filled]_ |
| ![img9](../images/IIHIM_-1057388368.jpg) | IIHIM_-1057388368.jpg | 11H(MATTHEW) | peter (0.8) ❌ | _[to be filled]_ |
| ![img10](../images/IIHIM_RIJKS_1878098591.jpg) | IIHIM_RIJKS_1878098591.jpg | 11HH(CATHERINE) | catherine (0.7) | _[to be filled]_ |

## Analysis Notes

### API Performance Summary
- **Batch 1**: 5/5 correct (100%)
- **Batch 2**: 3/5 correct (60%)
- **Overall**: 8/10 correct (80%)

### Key Observations
- **Perfect API predictions** in Batch 1 - all saints correctly identified
- **Two API errors** in Batch 2:
  - Image 8: Expected Catherine → Predicted Mary Magdalene
  - Image 9: Expected Matthew → Predicted Peter
- **High confidence scores** for most predictions (0.6-0.9 range)

### Comparison Instructions
1. Fill in ChatGPT predictions in the tables above
2. Compare ChatGPT results with API predictions
3. Note any differences in classification or confidence
4. Analyze patterns in errors or successes
5. Document any systematic differences between API and web interface