# Wikidata

With the following SPARQL query I downloaded the paintings, from Wikidata, with at least one ICONCLASS class related, starting with 11H or 11HH

```
SELECT ?painting ?image ?iconclass WHERE {
  ?painting wdt:P31 wd:Q3305213;        
           wdt:P1257 ?iconclass.        
  ?painting wdt:P18 ?image.            
  FILTER(strstarts(?iconclass, '11H'))
}
```

The result, for paintings with single label, is 724 images with these classes:

```
11HH(MARY MAGDALENE)     177
11H(JOHN THE BAPTIST)    131
11H(JEROME)               78
11HH(CATHERINE)           76
11H(PETER)                68
11H(JOHN)                 51
11H(FRANCIS)              40
11H(ANTONY ABBOT)         38
11H(JOSEPH)               35
11H(PAUL)                 31
```

With these data:
| ID                     | Label                 | Description                                                                                               |
|------------------------|----------------------|-----------------------------------------------------------------------------------------------------------|
| 11HH(MARY MAGDALENE)  | Mary Magdalene       | the penitent harlot Mary Magdalene; possible attributes: book (or scroll), crown, crown of thorns, crucifix, jar of ointment, mirror, musical instrument, palm-branch, rosary, scourge |
| 11H(JOHN THE BAPTIST) | St. John the Baptist | John the Baptist; possible attributes: book, reed cross, baptismal cup, honeycomb, lamb, staff            |
| 11H(JEROME)           | St. Jerome           | the monk and hermit Jerome (Hieronymus); possible attributes: book, cardinal's hat, crucifix, hour-glass, lion, skull, stone |
| 11HH(CATHERINE)       | St. Catherine        | the virgin martyr Catherine of Alexandria; possible attributes: book, crown, emperor Maxentius, palm-branch, ring, sword, wheel |
| 11H(PETER)            | St. Peter            | the apostle Peter, first bishop of Rome; possible attributes: book, cock, (upturned) cross, (triple) crozier, fish, key, scroll, ship, tiara |
| 11H(JOHN)             | St. John the Evangelist | the apostle John the Evangelist; possible attributes: book, cauldron, chalice with snake, eagle, palm, scroll |
| 11H(PAUL)             | St. Paul             | the apostle Paul of Tarsus; possible attributes: book, scroll, sword                                      |
| 11H(ANTONY ABBOT)     | St. Anthony Abbot    | the hermit Antony Abbot (Antonius Abbas) of Egypt, also called the Great; possible attributes: bell, book, T-shaped staff, flames, pig |
| 11H(FRANCIS)          | St. Francis of Assisi | founder of the Order of Friars Minor (Franciscans), Francis(cus) of Assisi; possible attributes: book, crucifix, lily, skull, stigmata |
| 11H(JOSEPH)           | St. Joseph           | the foster-father of Christ, Joseph of Nazareth, husband of Mary; possible attributes: flowering rod or wand, lily, carpenter's tools |