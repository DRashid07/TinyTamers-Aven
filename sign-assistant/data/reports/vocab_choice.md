# Vocabulary and group split

This report records the original 40-class selection. The recognition vocabulary was subsequently
extended with 20 everyday classes while retaining these class indices and the original group split.
See [vocab_expansion.md](vocab_expansion.md) for the extension and its train/validation counts.

Written by `python -m data.choose_vocab` (owner A). No model results were used.

Settings: camera=any, min videos=25, min groups=6, max classes=40, seeds tried=200, group column=group.
Index: 7248 'any' videos in 100 classes. 40 classes chosen.
AzSLD has no signer IDs: a group is the recording date of the Sentences video a clip was cut from (a proxy, so one signer can be in train and val). The groups columns count those dates.

## Candidates (priority order)

| # | word | dataset_label | videos | groups | status |
|---|---|---|---|---|---|
| 1 | mən | MƏN | 2527 | 59 | kept |
| 2 | sən | SƏN | 48 | 26 | kept |
| 3 | həkim | - | 0 | 0 | not in dataset |
| 4 | xəstəxana | - | 0 | 0 | not in dataset |
| 5 | getmək | GETMƏK | 48 | 20 | kept |
| 6 | istəmək | İSTƏMƏK | 105 | 27 | kept |
| 7 | ağrı | - | 0 | 0 | not in dataset |
| 8 | sabah | SABAH | 43 | 23 | kept |
| 9 | bu gün | BU GÜN | 35 | 23 | kept |
| 10 | vaxt | VAXT | 24 | 19 | too few videos |
| 11 | kömək | - | 0 | 0 | not in dataset |
| 12 | sənəd | SƏNƏD | 60 | 39 | kept |

Missing from the dataset: həkim, xəstəxana, ağrı, kömək.

## Vocabulary (class index = row order)

| index | id | gloss | az | dataset_label | videos | groups | why |
|---|---|---|---|---|---|---|---|
| 0 | men | MƏN | mən | MƏN | 2527 | 59 | candidate 'mən' |
| 1 | sen | SƏN | sən | SƏN | 48 | 26 | candidate 'sən' |
| 2 | getmek | GETMƏK | getmək | GETMƏK | 48 | 20 | candidate 'getmək' |
| 3 | istemek | İSTƏMƏK | istəmək | İSTƏMƏK | 105 | 27 | candidate 'istəmək' |
| 4 | sabah | SABAH | sabah | SABAH | 43 | 23 | candidate 'sabah' |
| 5 | bu_gun | BU GÜN | bu gün | BU GÜN | 35 | 23 | candidate 'bu gün' |
| 6 | sened | SƏNƏD | sənəd | SƏNƏD | 60 | 39 | candidate 'sənəd' |
| 7 | siz | SİZ | siz | SİZ | 440 | 50 | fill: most videos/groups |
| 8 | bu | BU | bu | BU | 334 | 47 | fill: most videos/groups |
| 9 | o | O | o | O | 303 | 48 | fill: most videos/groups |
| 10 | menim | MƏNİM | mənim | MƏNİM | 168 | 24 | fill: most videos/groups |
| 11 | biz | BİZ | biz | BİZ | 167 | 34 | fill: most videos/groups |
| 12 | onun | ONUN | onun | ONUN | 118 | 24 | fill: most videos/groups |
| 13 | elil | ƏLİL | əlil | ƏLİL | 111 | 32 | fill: most videos/groups |
| 14 | var | VAR | var | VAR | 110 | 34 | fill: most videos/groups |
| 15 | burda | BURDA | burda | BURDA | 102 | 42 | fill: most videos/groups |
| 16 | ev | EV | ev | EV | 77 | 38 | fill: most videos/groups |
| 17 | usaq | UŞAQ | uşaq | UŞAQ | 69 | 17 | fill: most videos/groups |
| 18 | hansi | HANSI | hansı | HANSI | 68 | 39 | fill: most videos/groups |
| 19 | pensiya | PENSİYA | pensiya | PENSİYA | 67 | 18 | fill: most videos/groups |
| 20 | baki | BAKI | bakı | BAKI | 64 | 33 | fill: most videos/groups |
| 21 | d | D | d | D | 55 | 25 | fill: most videos/groups |
| 22 | saglam | SAĞLAM | sağlam | SAĞLAM | 55 | 21 | fill: most videos/groups |
| 23 | biler | BİLƏR | bilər | BİLƏR | 49 | 23 | fill: most videos/groups |
| 24 | futbol | FUTBOL | futbol | FUTBOL | 46 | 23 | fill: most videos/groups |
| 25 | mene | MƏNƏ | mənə | MƏNƏ | 46 | 18 | fill: most videos/groups |
| 26 | almaq | ALMAQ | almaq | ALMAQ | 45 | 21 | fill: most videos/groups |
| 27 | etibarname | ETİBARNAMƏ | etibarnamə | ETİBARNAMƏ | 43 | 22 | fill: most videos/groups |
| 28 | yox | YOX | yox | YOX | 42 | 17 | fill: most videos/groups |
| 29 | bilmir | BİLMİR | bilmir | BİLMİR | 41 | 18 | fill: most videos/groups |
| 30 | olmaq | OLMAQ | olmaq | OLMAQ | 40 | 21 | fill: most videos/groups |
| 31 | telefon | TELEFON | telefon | TELEFON | 40 | 20 | fill: most videos/groups |
| 32 | dunen | DÜNƏN | dünən | DÜNƏN | 39 | 31 | fill: most videos/groups |
| 33 | is | İŞ | iş | İŞ | 39 | 26 | fill: most videos/groups |
| 34 | viza | VİZA | viza | VİZA | 38 | 15 | fill: most videos/groups |
| 35 | orda | ORDA | orda | ORDA | 37 | 29 | fill: most videos/groups |
| 36 | azerbaycan | AZƏRBAYCAN | azərbaycan | AZƏRBAYCAN | 37 | 24 | fill: most videos/groups |
| 37 | harda | HARDA | harda | HARDA | 36 | 21 | fill: most videos/groups |
| 38 | yemek | YEMƏK | yemək | YEMƏK | 36 | 18 | fill: most videos/groups |
| 39 | ana | ANA | ana | ANA | 35 | 19 | fill: most videos/groups |

Gloss differs from dataset_label only for known spelling slips in the dataset: QARABAQ -> QARABAĞ, ÜNVANLİ -> ÜNVANLI, EŞİTMƏ MƏHDÜDİYYƏTLİ -> EŞİTMƏ MƏHDUDİYYƏTLİ.

## Group split (CONTRACT v2)

Dataset groups go to train/val; test = team recordings: EMPTY, no team recordings (team_* groups) in index.csv yet. Seed 52: the smallest per-class video count in val is 6. 58 vocab videos have an empty group: train-only, not listed in splits.json.

| split | groups | videos | share |
|---|---|---|---|
| train | 54 | 4639 | 80% |
| val | 9 | 1166 | 20% |
| test | 0 | 0 | 0% |

Per class: videos (groups).

| id | gloss | train | val | test |
|---|---|---|---|---|
| men | MƏN | 1992 (50) | 513 (9) | 0 (0) |
| sen | SƏN | 42 (23) | 6 (3) | 0 (0) |
| getmek | GETMƏK | 38 (16) | 10 (4) | 0 (0) |
| istemek | İSTƏMƏK | 71 (22) | 34 (5) | 0 (0) |
| sabah | SABAH | 32 (18) | 11 (5) | 0 (0) |
| bu_gun | BU GÜN | 27 (18) | 8 (5) | 0 (0) |
| sened | SƏNƏD | 49 (32) | 11 (7) | 0 (0) |
| siz | SİZ | 357 (41) | 83 (9) | 0 (0) |
| bu | BU | 259 (41) | 54 (6) | 0 (0) |
| o | O | 239 (43) | 49 (5) | 0 (0) |
| menim | MƏNİM | 139 (20) | 29 (4) | 0 (0) |
| biz | BİZ | 135 (27) | 32 (7) | 0 (0) |
| onun | ONUN | 99 (19) | 19 (5) | 0 (0) |
| elil | ƏLİL | 93 (25) | 18 (7) | 0 (0) |
| var | VAR | 86 (28) | 24 (6) | 0 (0) |
| burda | BURDA | 82 (35) | 20 (7) | 0 (0) |
| ev | EV | 62 (32) | 15 (6) | 0 (0) |
| usaq | UŞAQ | 49 (14) | 20 (3) | 0 (0) |
| hansi | HANSI | 51 (31) | 17 (8) | 0 (0) |
| pensiya | PENSİYA | 55 (16) | 12 (2) | 0 (0) |
| baki | BAKI | 53 (28) | 11 (5) | 0 (0) |
| d | D | 44 (21) | 11 (4) | 0 (0) |
| saglam | SAĞLAM | 45 (18) | 10 (3) | 0 (0) |
| biler | BİLƏR | 39 (19) | 10 (4) | 0 (0) |
| futbol | FUTBOL | 34 (17) | 12 (6) | 0 (0) |
| mene | MƏNƏ | 40 (16) | 6 (2) | 0 (0) |
| almaq | ALMAQ | 34 (17) | 11 (4) | 0 (0) |
| etibarname | ETİBARNAMƏ | 35 (19) | 8 (3) | 0 (0) |
| yox | YOX | 31 (12) | 11 (5) | 0 (0) |
| bilmir | BİLMİR | 30 (13) | 11 (5) | 0 (0) |
| olmaq | OLMAQ | 29 (17) | 11 (4) | 0 (0) |
| telefon | TELEFON | 29 (16) | 11 (4) | 0 (0) |
| dunen | DÜNƏN | 32 (26) | 7 (5) | 0 (0) |
| is | İŞ | 32 (21) | 7 (5) | 0 (0) |
| viza | VİZA | 29 (12) | 9 (3) | 0 (0) |
| orda | ORDA | 29 (23) | 8 (6) | 0 (0) |
| azerbaycan | AZƏRBAYCAN | 31 (19) | 6 (5) | 0 (0) |
| harda | HARDA | 29 (17) | 7 (4) | 0 (0) |
| yemek | YEMƏK | 28 (14) | 8 (4) | 0 (0) |
| ana | ANA | 29 (14) | 6 (5) | 0 (0) |
