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

to re-execute