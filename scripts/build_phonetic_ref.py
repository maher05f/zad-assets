"""النص الصوتي المرجعي لكل الآيات (6236) بـ quran_phonetizer من quran-transcript.

    <venv>/python scripts/build_phonetic_ref.py --model src/muaalem-model-v3_2 \
        --muaalem-src src/quran-muaalem/src --out out/phonetic_ref.json

لكل آية:
  phonemes  النص الصوتي كما يتوقعه المعلّم القرآني (remove_spaces=True)
  groups    مجموعات الأحرف التي تُنسب إليها الصفات (ملتصقة تساوي phonemes)
  sifat     لكل مستوى نص أرقام بطول groups: رقم كل صفة في مفردات ذلك المستوى في النموذج
            (muaalem-levels.json → idToToken)، فيُقارن مباشرة بمخرجات النموذج
"""
import argparse, json, sys, time
from importlib.metadata import distribution, version
from typing import Literal, get_origin

from quran_transcript import Aya, MoshafAttributes, SifaOutput, quran_phonetizer

# رواية حفص عن عاصم من طريق الشاطبية، المدود ٤ حركات (يمكن تغييرها من سطر الأوامر)
MOSHAF = dict(rewaya='hafs', madd_monfasel_len=4, madd_mottasel_len=4, madd_mottasel_waqf=4, madd_aared_len=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--muaalem-src', required=True)
    for k, v in MOSHAF.items():
        if k != 'rewaya':
            ap.add_argument('--' + k.replace('_', '-'), type=int, default=v)
    a = ap.parse_args()
    sys.path.insert(0, a.muaalem_src)
    from quran_muaalem.modeling.vocab import SIFAT_ATTR_TO_ARABIC
    from quran_muaalem.modeling.multi_level_tokenizer import MultiLevelTokenizer

    tok = MultiLevelTokenizer(a.model)
    levels = [k for k, f in SifaOutput.model_fields.items() if get_origin(f.annotation) == Literal]
    code = {}
    for lv in levels:
        tok_to_id = {t: i for i, t in tok.id_to_vocab[lv].items()}
        code[lv] = {name: str(tok_to_id[ar]) for name, ar in SIFAT_ATTR_TO_ARABIC.items() if ar in tok_to_id}
    assert all(len(v) == 1 for c in code.values() for v in c.values())

    attrs = dict(MOSHAF, **{k: getattr(a, k) for k in MOSHAF if k != 'rewaya'})
    moshaf = MoshafAttributes(**attrs)
    ayat, t0 = [], time.time()
    for s in range(1, 115):
        n = Aya(s, 1).get().num_ayat_in_sura
        for i in range(1, n + 1):
            r = quran_phonetizer(Aya(s, i).get().uthmani, moshaf, remove_spaces=True, sura_idx=s)
            groups = [x.phonemes for x in r.sifat]
            assert ''.join(groups) == r.phonemes, (s, i)
            ayat.append({'s': s, 'a': i, 'phonemes': r.phonemes, 'groups': groups,
                         'sifat': {lv: ''.join(code[lv][getattr(x, lv)] for x in r.sifat) for lv in levels}})
        if s % 10 == 0 or s == 114:
            print('سورة %d %.0fs' % (s, time.time() - t0), flush=True)
    with open(a.out, 'w', encoding='utf-8') as fh:
        # النص مشتق من نص تنزيل العثماني: رخصته تشترط إرفاق كتلة حقوقه في كل ملف مشتق
        notice = distribution('quran-transcript').read_text('licenses/LICENSE')
        json.dump({'notice': notice, 'source': 'Tanzil Project — https://tanzil.net · quran-transcript (MIT) — https://github.com/obadx/quran-transcript',
                   'generator': 'quran-transcript %s quran_phonetizer(remove_spaces=True, sura_idx)' % version('quran-transcript'),
                   'moshaf': attrs, 'levels': levels,
                   'sifaCodes': {lv: {v: k for k, v in c.items()} for lv, c in code.items()},
                   'count': len(ayat), 'ayat': ayat}, fh, ensure_ascii=False, separators=(',', ':'))
    print('ok', len(ayat))


if __name__ == '__main__':
    main()
