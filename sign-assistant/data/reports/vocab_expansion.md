# Recognition vocabulary extension: 40 to 60 classes

Owner: A (Data/ML). Added 20 everyday classes with genuine AzSLD_Words_100 videos.
The original 40 class entries remain the prefix in their original order. The existing recording-date
train/val split is unchanged; no test data was used. Text-to-sign playback still has 100 classes.

All 497 additional landmark files for the shipped classes
were extracted successfully at target 15 fps. Training uses 5,080 clips;
validation uses 1,280 clips across all 60 classes.

| id | gloss | train | val | mean valid pose | mean hand presence |
|---|---|---:|---:|---:|---:|
| salam | SALAM | 22 | 8 | 100.0% | 93.1% |
| nece | NECƏ | 24 | 3 | 100.0% | 98.0% |
| yaxsi | YAXŞI | 12 | 7 | 100.0% | 97.6% |
| su | SU | 18 | 7 | 100.0% | 97.4% |
| ata | ATA | 17 | 4 | 100.0% | 96.3% |
| mekteb | MƏKTƏB | 19 | 6 | 100.0% | 94.0% |
| aile | AİLƏ | 17 | 8 | 100.0% | 98.0% |
| gelmek | GƏLMƏK | 27 | 7 | 100.0% | 95.7% |
| avtomobil | AVTOMOBİL | 27 | 5 | 100.0% | 92.5% |
| oxumaq | OXUMAQ | 19 | 8 | 100.0% | 99.3% |
| islemek | İŞLƏMƏK | 20 | 4 | 100.0% | 98.8% |
| vaxt | VAXT | 20 | 4 | 100.0% | 99.3% |
| axsam | AXŞAM | 21 | 6 | 100.0% | 96.0% |
| indi | İNDİ | 18 | 2 | 100.0% | 92.7% |
| cox | ÇOX | 19 | 4 | 100.0% | 98.7% |
| lazim | LAZIM | 17 | 11 | 100.0% | 92.0% |
| avtobus | AVTOBUS | 16 | 2 | 100.0% | 100.0% |
| qapi | QAPI | 10 | 8 | 100.0% | 94.8% |
| bu_hefte | BU HƏFTƏ | 27 | 5 | 100.0% | 95.4% |
| ad_gunu | AD GÜNÜ | 13 | 5 | 100.0% | 88.7% |

A small validation count is not evidence of accuracy on new signers. The groups are recording dates,
and the same signers can occur in training and validation. No unseen-signer test recordings are available.

The shipped data/vocab.json is the curated vocabulary; data.choose_vocab defaults to the original 40
selection. For retraining, extract and train using the shipped vocabulary instead of re-running that
selection command. See README.md for the data/model pipeline.
