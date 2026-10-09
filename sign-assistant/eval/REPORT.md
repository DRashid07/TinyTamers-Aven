# Evaluation report

Owner: D (Direction B/Speech/Eval). Written by `python -m eval.evaluate --split val|test|team`;
each split has its own section. The test split is run once with `--final` after the feature freeze (eval/test_runs.log).

<!-- BEGIN val -->
## Validation (val)

Date 2026-10-09 14:44 UTC · commit e3098b8 (+ uncommitted changes) · config sha256 ab702967ed3c · model gru, 60 classes, temperature 0.7558, tau 0.9, margin 0.2

Split: 1280 clips from 9 groups: rec_2022-05-31, rec_2022-06-16, rec_2022-07-13, rec_2022-07-21, rec_2022-07-29, rec_2022-08-02, rec_2022-11-02, rec_2022-11-19, rec_2022-12-20.
Clips without landmarks or with an unknown id (skipped): 0. segment_offline found no sign in 49 clips (diagnostic only: AzSLD clips are used whole, CONTRACT v2).

| metric | value |
|---|---|
| top-1 accuracy, no abstention | 69.5% |
| macro-accuracy (classes are imbalanced) | 81.5% |
| coverage at tau 0.9, margin 0.2 | 40.0% (512 of 1280) |
| selective accuracy at tau 0.9, margin 0.2 | 95.9% (491 of 512) |

![Row-normalised confusion matrix](figures/confusion_val.png)

Most confused pairs (symmetric rate = mean of the two row-normalised off-diagonal cells):

| pair | symmetric rate | counts |
|---|---|---|
| MƏNİM / MƏNƏ | 42.5% | MƏNİM→MƏNƏ 15 of 29, MƏNƏ→MƏNİM 2 of 6 |
| O / ORDA | 30.1% | O→ORDA 5 of 49, ORDA→O 4 of 8 |
| O / ONUN | 28.8% | O→ONUN 5 of 49, ONUN→O 9 of 19 |
| BİLMİR / AVTOBUS | 25.0% | BİLMİR→AVTOBUS 0 of 11, AVTOBUS→BİLMİR 1 of 2 |
| MƏN / MƏNİM | 20.3% | MƏN→MƏNİM 84 of 513, MƏNİM→MƏN 7 of 29 |

Coverage and selective accuracy by tau (margin 0.2):

| tau | coverage | selective accuracy | answered |
|---|---|---|---|
| 0.3 | 79.4% | 75.6% | 1016 |
| 0.4 | 78.7% | 75.6% | 1007 |
| 0.5 | 75.3% | 76.6% | 964 |
| 0.6 | 63.0% | 80.0% | 806 |
| 0.7 | 53.0% | 84.4% | 678 |
| 0.8 | 46.0% | 89.6% | 589 |
| 0.9 | 40.0% | 95.9% | 512 |
| 0.95 | 34.1% | 97.9% | 437 |

Per-class accuracy (no abstention):

| gloss | clips | top-1 |
|---|---|---|
| MƏN | 513 | 52.8% |
| SƏN | 6 | 66.7% |
| GETMƏK | 10 | 50.0% |
| İSTƏMƏK | 34 | 97.1% |
| SABAH | 11 | 72.7% |
| BU GÜN | 8 | 100.0% |
| SƏNƏD | 11 | 100.0% |
| SİZ | 83 | 84.3% |
| BU | 54 | 92.6% |
| O | 49 | 67.3% |
| MƏNİM | 29 | 20.7% |
| BİZ | 32 | 71.9% |
| ONUN | 19 | 5.3% |
| ƏLİL | 18 | 94.4% |
| VAR | 24 | 100.0% |
| BURDA | 20 | 90.0% |
| EV | 15 | 100.0% |
| UŞAQ | 20 | 100.0% |
| HANSI | 17 | 82.4% |
| PENSİYA | 12 | 91.7% |
| BAKI | 11 | 100.0% |
| D | 11 | 72.7% |
| SAĞLAM | 10 | 100.0% |
| BİLƏR | 10 | 90.0% |
| FUTBOL | 12 | 75.0% |
| MƏNƏ | 6 | 66.7% |
| ALMAQ | 11 | 72.7% |
| ETİBARNAMƏ | 8 | 100.0% |
| YOX | 11 | 81.8% |
| BİLMİR | 11 | 90.9% |
| OLMAQ | 11 | 90.9% |
| TELEFON | 11 | 100.0% |
| DÜNƏN | 7 | 85.7% |
| İŞ | 7 | 85.7% |
| VİZA | 9 | 88.9% |
| ORDA | 8 | 37.5% |
| AZƏRBAYCAN | 6 | 83.3% |
| HARDA | 7 | 57.1% |
| YEMƏK | 8 | 87.5% |
| ANA | 6 | 100.0% |
| SALAM | 8 | 87.5% |
| NECƏ | 3 | 100.0% |
| YAXŞI | 7 | 85.7% |
| SU | 7 | 71.4% |
| ATA | 4 | 75.0% |
| MƏKTƏB | 6 | 83.3% |
| AİLƏ | 8 | 87.5% |
| GƏLMƏK | 7 | 71.4% |
| AVTOMOBİL | 5 | 80.0% |
| OXUMAQ | 8 | 100.0% |
| İŞLƏMƏK | 4 | 100.0% |
| VAXT | 4 | 100.0% |
| AXŞAM | 6 | 50.0% |
| İNDİ | 2 | 100.0% |
| ÇOX | 4 | 75.0% |
| LAZIM | 11 | 81.8% |
| AVTOBUS | 2 | 50.0% |
| QAPI | 8 | 87.5% |
| BU HƏFTƏ | 5 | 100.0% |
| AD GÜNÜ | 5 | 100.0% |

**What this means.** Without abstaining, the model names the right sign for 70% of these clips; counting every sign equally (macro) it is 82%. The worst mix-up is MƏNİM / MƏNƏ. Top-1 is lower than macro because most errors come from one big class: MƏN is wrong in 242 of 513 clips, mostly taken for MƏNƏ. With the abstain rule (tau 0.9, margin 0.2) it answers 40% of the clips and says "Əmin deyiləm" for the rest; when it answers it is right 96% of the time. Caution: val groups are recording dates of the same AzSLD signers that are in train, so these numbers are optimistic for a new signer; only the test and team sections measure that.
<!-- END val -->
