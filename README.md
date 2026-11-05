# zad-assets

الملفات الكبيرة لتطبيق **زاد**: نماذج التسميع وحزم البيانات. التطبيق يقرأ
[`assets.json`](assets.json) وينزّل ما يحتاجه منها عند الطلب، ويتحقق من الحجم و SHA-256 قبل الاستعمال.

```
https://raw.githubusercontent.com/maher05f/zad-assets/main/assets.json
```

الملفات نفسها في [Releases](https://github.com/maher05f/zad-assets/releases)، لا في المستودع.

## الإصدارات

### `models-v1` — نماذج التسميع

| المعرّف | الملف | الأصل | الرخصة |
|---|---|---|---|
| `model-light` | `ggml-quran-base-q5_0.bin` (whisper.cpp) | [tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) — Tarteel AI | Apache-2.0 |
| `model-strong` | `muaalem-int8.onnx` + `muaalem-levels.json` + `phonetic_ref.json` | [obadx/muaalem-model-v3_2](https://huggingface.co/obadx/muaalem-model-v3_2) — [المعلّم القرآني](https://github.com/obadx/quran-muaalem) | QML 1.0 |
| `model-full` | `muaalem-fp32.onnx` + `muaalem-fp32.onnx.data` (جزءان) + الملفان نفسهما | المعلّم القرآني نفسه بدقة fp32 كاملة دون ضغط | QML 1.0 |

النسخ الثلاث يختارها التطبيق حسب ذاكرة الهاتف (`minRamGb`). `model-strong` و`model-full` لهما المدخلات
والمخرجات وترتيب المستويات نفسها، فيعمل معهما الكود نفسه. ملف أوزان `model-full` أكبر من حد GitHub (2GB)،
فهو مقسوم إلى `muaalem-fp32.onnx.data.part1` و`.part2`: تُدمج بالترتيب في `muaalem-fp32.onnx.data`
بجانب `muaalem-fp32.onnx`، ويُتحقق من SHA-256 لكل جزء وللملف بعد الدمج.

**رخصة QML 1.0 وملف NOTICE يشملان كل نسخ المعلّم القرآني هنا (int8 و fp32).**

> هذا المعلّم القرآني نموذج ذكاء اصطناعي قد يخطئ؛ فهو للإعانة فقط ولا يُعتمد عليه، وليس بديلاً عن معلّمٍ مُتقِنٍ مُجاز؛ لأنّ القرآن لا يُؤخذ إلا بالمشافهة.

رخصة QML تُلزم كل واجهة مبنية على `model-strong` بعرض هذا التنبيه قبل أول استعمال، وإبقائه
في متناول المستخدم من القائمة أو صفحة «حول»، بالعربية وبكل لغة في الواجهة؛ وتمنع الإعلانات.
التفاصيل في [`licenses/NOTICE-muaalem.txt`](licenses/NOTICE-muaalem.txt) و[`licenses/LICENSE-muaalem-QML-1.0.txt`](licenses/LICENSE-muaalem-QML-1.0.txt).

- `muaalem-int8.onnx` يقبل الصوت الخام: `waveform` float32 بالشكل `[1, N]`، 16kHz، قناة واحدة، بين -1 و1.
  حساب fbank (80 مِل، تطبيع لكل مِل، تكديس إلى 160) داخل النموذج. المخرجات: `logits_<level>` لكل مستوى.
- الذاكرة تكبر بمربع طول المقطع (انتباه بمواضع نسبية)، فيُمرَّر الصوت مقاطع لا تتجاوز ~15 ثانية
  (مثلاً آية آية). القياس على Windows (4 خيوط، أقصى ذاكرة فعلية):

  | طول المقطع | int8 | fp32 |
  |---|---|---|
  | 5 ث | 0.8 GB | 2.4 GB |
  | 10 ث | 1.0 GB | 2.6 GB |
  | 15 ث | 1.3 GB | 3.0 GB |
  | 20 ث | 1.8 GB | 3.4 GB |
  | 30 ث | 3.1 GB | 4.7 GB |
- `muaalem-levels.json`: ترتيب المخرجات ومفردات كل مستوى، ورقم الفراغ (blank) لفك CTC.
- `phonetic_ref.json`: النص الصوتي المرجعي لكل آية (6236) بـ `quran_phonetizer`، رواية حفص، المدود ٤ حركات.
  مشتق من نص [Tanzil](https://tanzil.net)، وكتلة حقوقه داخل الملف وفي `licenses/LICENSE-phonetic-ref.txt`.

### `data-v1` — حزم البيانات (zip)

كل حزمة فيها `LICENSE` و`SOURCE.md`، والملفات داخلها كما هي من مصادرها دون تعديل.

| الحزمة | المحتوى | المصدر والرخصة |
|---|---|---|
| `mushaf-qpc2-base` | تخطيط الصفحات الـ604 + خط البسملة QCF2BSML | خطوط [مجمع الملك فهد لطباعة المصحف الشريف](https://qurancomplex.gov.sa) غير معدّلة · التخطيط [zonetecde/mushaf-layout](https://github.com/zonetecde/mushaf-layout) (ISC) |
| `mushaf-qpc2-juz01` … `juz30` | خطوط صفحات كل جزء | مجمع الملك فهد، غير معدّلة |
| `font-uthmanic-hafs` | خط حفص العثماني v22 + نصه | مجمع الملك فهد (رخصة الخط المضمَّنة: الاستعمال والنسخ والتوزيع مجاناً، دون بيع أو تعديل) |
| `quranpedia-core` | التصريف، المعاني، علوم السور، الكتب، الناسخ والمنسوخ، المتشابه، الموضوعات | [Quranpedia.net](https://quranpedia.net) — نسخة البيانات 2026-11-03 |
| `quranpedia-e3rab` | الإعراب (4 كتب) + شجرة الإعراب | Quranpedia.net · [The Quranic Treebank](https://github.com/NoorBayan/Quranic) (MIT) |
| `quranpedia-qiraat` | القراءات + 13 رواية | Quranpedia.net |
| `quranpedia-asbab` | أسباب النزول | Quranpedia.net |
| `corpus-morphology` | الصرف كلمة بكلمة + ترجمة الوسوم | [Quranic Arabic Corpus](https://corpus.quran.com) (GPL-3.0)، نسخة [mustafa0x/quran-morphology](https://github.com/mustafa0x/quran-morphology) |

بيانات Quranpedia تُصحَّح باستمرار؛ للتحديث: `https://quranpedia.net/api/v1/changes?since=2026-11-03`.

## `assets.json`

```json
{"id": "model-strong", "type": "model", "version": 1,
 "url": "https://github.com/maher05f/zad-assets/releases/download/models-v1/muaalem-int8.onnx",
 "size": 0, "sha256": "…", "minRamGb": 6,
 "extra":   [{"file": "muaalem-levels.json", "url": "…", "size": 0, "sha256": "…"}],
 "license": [{"file": "NOTICE-muaalem.txt",  "url": "…", "size": 0, "sha256": "…"}]}
```

حزم Quranpedia فيها أيضاً `"sourceVersion": "2026-11-03"`.

## إعادة البناء

السكربتات في [`scripts/`](scripts):

| السكربت | ما يفعله |
|---|---|
| `build_packs.py` | يبني حزم `data-v1` من مجلد البيانات المحلي (حتمي: نفس الملفات ← نفس SHA-256) |
| `convert_light.py` | Whisper ترتيل → GGML ثم q5_0، ويجرّبه بـ whisper-cli |
| `export_strong.py` | المعلّم القرآني → ONNX (opset 17) مع fbank داخله → int8، ويقارن بـ PyTorch، ويحفظ ملف اختبار |
| `build_phonetic_ref.py` | `phonetic_ref.json` لكل الآيات |
| `compare_models.py` | يقارن نسخ ONNX بـ PyTorch مستوى بمستوى (الحروف، الحركات، كل صفة) |
| `measure_ram.py` | أقصى استهلاك للذاكرة لكل نسخة على ملف صوتي |
| `split_parts.py` | يقسم ملفاً أكبر من 1.9 GiB إلى أجزاء، و`--merge` يدمجها |
| `make_assets.py` | يحسب الحجم و SHA-256 ويكتب `assets.json`، و`--verify` ينزّل كل رابط ويتحقق منه (والمقسوم بعد الدمج) |

نصوص الرخص كاملة في [`licenses/`](licenses).
