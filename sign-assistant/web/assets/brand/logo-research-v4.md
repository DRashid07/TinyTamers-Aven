<!-- Owner: Codex (Brand/logo assets). -->
# AzSL — sıfırdan hazırlanmış yazısız loqo

Tarix: 9 oktyabr 2026.
Konsept: **Ünsiyyət körpüsü**.
[Son PNG](azsl-communication-bridge-v4.png) · [Təqdimat önizləməsi](logo-preview-v4.html) · [Tam imagegen promptları](imagegen-prompts-v4.txt).

## Layihə ilə əlaqə

Layihənin [README](../../../README.md) və [CONTRACT](../../../CONTRACT.md) faylları yenidən araşdırılıb. Bu, Azərbaycan işarə dili üçün köməkçi prototipdir: bir istiqamətdə kamera jestlərindən işarələri tanımaq, digər istiqamətdə mətni işarə videolarına çevirmək məqsədi daşıyır.

Yeni simvolda iki balanslı açıq yol işarə və mətn/səs ünsiyyətini təmsil etmək üçün seçilib. Aralarındakı ortaq keçid qarşılıqlı anlaşma və ünsiyyətə çıxış ideyasını daşıyır. Bu, dizaynın məqsədli metaforasıdır.

## Araşdırma və qərarlar

| Mənbə | Mənbədə vurğulanan fikir | Bu loqo üçün çıxardığım qərar |
| --- | --- | --- |
| [Azercell — məqsəd və dəyərlər](https://www.azercell.com/en/about-us/deyerlerimiz.html) | Azərbaycanda hər kəsin həyatını bağlantı vasitəsilə gücləndirmək; birlikdə işləmək; asan istifadə və etibar | İki tərəfə bərabər vizual çəki vermək və mərkəzdə geniş açıq sahə saxlamaq |
| [Azercell — brend kimliyi](https://www.azercell.com/az/about-us/press-releases/news/biz-dyiirik-ki-hyatlari-dyik.html) | Hərəkət, inkişaf və innovasiya | Qarşılıqlı axın hissi yaradan açıq, yumşaq həndəsə |
| [Azercell — 2024 hesabatı](https://www.azercell.com/assets/files/sustainability/report_2024_aze_02.pdf) | Rəsmi materiallarda bənövşəyi vizual mühit | Azercell-lə vizual uyğunluq üçün dərin bənövşəyi istiqamət |
| [Pentagram — Wings](https://www.pentagram.com/work/wings) | Daha premium mövqeləndirmə; məqsədli daxili boşluq; fərqlilik və brend ailəsi ilə əlaqə | Formanı və daxili boşluğu bir konseptdən qurmaq, öz konturunu saxlamaq |
| [Pentagram — Rolls-Royce](https://www.pentagram.com/work/rolls-royce-3/story) | Müxtəlif ölçülərə uyğun diqqətlə işlənmiş simvol və sakit, güclü vizual sistem | Dəqiq kontur, az detal, geniş nəfəs sahəsi |
| [Pentagram — Mytheresa](https://www.pentagram.com/work/mytheresa) | Sadəlik, aydınlıq və rəqəmsal istifadə | Kiçik ölçüdə də aydın qala bilən iki əsas forma |

Sağ sütundakı nəticələr mənim dizayn şərhimdir. Premium təsir üçün seçilən istiqamət dəqiq nisbətlər, az detal və sakit vizual çəkidir.

## Yeni forma

Sıfırdan qurulmuş iki açıq, qarşılıqlı lent dirsəyi var. Birinci forma sol tərəfdən qalxaraq yuxarıda mərkəzə yönəlir; ikinci forma sağ tərəfdən enərək aşağıda mərkəzə yönəlir. Ortada açıq keçid qalır.

Yazısız görünüş saxlanılıb. Yeni dizaynın əsas vizual dili insan əlaqəsini abstrakt həndəsə ilə ifadə edir. Azercell uyğunluğu rəng, axıcılıq və bağlantı ideyası üzərindən qurulub. Seçilmiş bənövşəyi layihə tonudur; rəsmi rəng spesifikasiyası kimi təqdim edilmir.

## Fayl və istifadə

- `azsl-communication-bridge-v4.png` — şəffaf PNG, **1254 × 1254 px**.
- `logo-preview-v4.html` — açıq fonda əsas loqo, tünd fonda ağ tətbiq nümunəsi və kiçik ölçülər.
- `imagegen-prompts-v4.txt` — ilk sıfırdan generasiya və son həndəsi dəqiqləşdirmə promptları.

PNG-ni təqdimata birbaşa yerləşdirmək olar. Nisbət 1:1-dir. Loqo ətrafındakı boşluğu saxlayın.

```html
<img src="assets/brand/azsl-communication-bridge-v4.png"
     alt="AzSL — Ünsiyyət körpüsü" width="64" height="64">
```

## Hazırlanma və yoxlama

Daxili `image_gen.imagegen` aləti istifadə edilib. İlk generasiyaya əvvəlki loqolar istinad şəkli kimi verilməyib; yeni konsept sıfırdan yaradılıb. İkinci mərhələdə daxili konturların keçidləri dəqiqləşdirilib. Son fayl dəyişdirilmədən layihəyə köçürülüb.

PNG-nin ölçüləri və ARGB formatı yoxlanılıb. Dörd künc və mərkəzi keçiddə alfa 0-dır. Mənbə ilə layihədəki nüsxənin SHA-256 heşləri eynidir. Önizləmədə tünd fon üzərində ağ rəng yalnız göstərilmə üslubudur; əsas PNG bənövşəyidir.

