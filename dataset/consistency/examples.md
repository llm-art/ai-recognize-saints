## Example Image Pairs for gpt-4o (test_3)

### Correctly Predicted Pairs

| Pair | Image 1 | Dataset | Ground Truth | Predicted | Image 2 | Dataset | Ground Truth | Predicted |
|------|---------|---------|--------------|-----------|---------|---------|--------------|-----------|
| 1 | ![Image 1](dataset/consistency/example/correct_1/image_1_1939_1_291.jpg) | ArtDL | 11H(JOHN THE BAPTIST) | 11H(JOHN THE BAPTIST) | ![Image 2](dataset/consistency/example/correct_1/image_2_Q20173065.jpg) | wikidata | 11H(JOHN THE BAPTIST) | 11H(JOHN THE BAPTIST) |
| 2 | ![Image 1](dataset/consistency/example/correct_2/image_1_1939_1_80.jpg) | ArtDL | 11H(PETER) | 11H(PETER) | ![Image 2](dataset/consistency/example/correct_2/image_2_Q20173671.jpg) | wikidata | 11H(PETER) | 11H(PETER) |
| 3 | ![Image 1](dataset/consistency/example/correct_3/image_1_1950_11_1_a.jpg) | ArtDL | 11H(PETER) | 11H(PETER) | ![Image 2](dataset/consistency/example/correct_3/image_2_Q20173413.jpg) | wikidata | 11H(PETER) | 11H(PETER) |
| 4 | ![Image 1](dataset/consistency/example/correct_4/image_1_253141.jpg) | ArtDL | 11H(JEROME) | 11H(JEROME) | ![Image 2](dataset/consistency/example/correct_4/image_2_Q3947314.jpg) | wikidata | 11H(JEROME) | 11H(JEROME) |
| 5 | ![Image 1](dataset/consistency/example/correct_5/image_1_253669.jpg) | ArtDL | 11HH(MARY MAGDALENE) | 11HH(MARY MAGDALENE) | ![Image 2](dataset/consistency/example/correct_5/image_2_Q20540321.jpg) | wikidata | 11HH(MARY MAGDALENE) | 11HH(MARY MAGDALENE) |

### Incorrectly Predicted Pairs

| Pair | Image 1 | Dataset | Ground Truth | Predicted | Image 2 | Dataset | Ground Truth | Predicted |
|------|---------|---------|--------------|-----------|---------|---------|--------------|-----------|
| 1 | ![Image 1](dataset/consistency/example/incorrect_1/image_1_ICCD3163621_13815-H.jpg) | ArtDL | 11F(MARY) | 11F(MARY) | ![Image 2](dataset/consistency/example/incorrect_1/image_2_IIHIM_RIJKS_1401436342.jpg) | ICONCLASS | 11HH(MARY MAGDALENE) | 11HH(MARY MAGDALENE) |
| 2 | ![Image 1](dataset/consistency/example/incorrect_2/image_1_ICCD3710537_375754.jpg) | ArtDL | 11F(MARY) | 11HH(MARY MAGDALENE) | ![Image 2](dataset/consistency/example/incorrect_2/image_2_IIHIM_RIJKS_1827277148.jpg) | ICONCLASS | 11HH(CATHERINE) | 11H(FRANCIS) |
| 3 | ![Image 1](dataset/consistency/example/incorrect_3/image_1_Q29477236.jpg) | ArtDL | 11HH(MARY MAGDALENE) | 11F(MARY) | ![Image 2](dataset/consistency/example/incorrect_3/image_2_Q29477236.jpg) | wikidata | 11HH(MARY MAGDALENE) | 11H(JOHN THE BAPTIST) |
| 4 | ![Image 1](dataset/consistency/example/incorrect_4/image_1_IIHIM_RIJKS_-649904531.jpg) | ICONCLASS | 11H(JEROME) | 11HH(MARY MAGDALENE) | ![Image 2](dataset/consistency/example/incorrect_4/image_2_Q17328232.jpg) | wikidata | 11H(JEROME) | 11H(JOSEPH) |
