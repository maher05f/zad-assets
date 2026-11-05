"""يقارن نسخ ONNX من المعلّم القرآني بـ PyTorch، مستوى بمستوى.

    <venv>/python scripts/compare_models.py --work <work-dir> out/muaalem-fp32.onnx out/muaalem-int8.onnx

المرجع: PyTorch fp32 مع SeamlessM4TFeatureExtractor الأصلي. لكل مستوى يُفك CTC (greedy)
ويُقاس التطابق = 1 − مسافة التحرير ÷ طول المرجع، مجموعاً على كل الملفات.
مستوى phonemes يُقسم إلى: الحروف (ومعها علامات المد) والحركات (َ ُ ِ وبقية العلامات غير المتباعدة).
ويُذكر أيضاً تطابق الإطارات (argmax لكل إطار).
"""
import argparse, glob, json, os, sys, unicodedata

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_strong import ctc_greedy, edit_distance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--json')
    ap.add_argument('models', nargs='+')
    a = ap.parse_args()
    w = a.work
    sys.path.insert(0, os.path.join(w, 'src', 'quran-muaalem', 'src'))
    import onnxruntime as ort
    from transformers import AutoFeatureExtractor
    from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
    from quran_muaalem.modeling.multi_level_tokenizer import MultiLevelTokenizer
    from quran_muaalem.modeling.vocab import PAD_TOKEN_IDX

    mdir = os.path.join(w, 'src', 'muaalem-model-v3_2')
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(mdir).eval().float()
    ext = AutoFeatureExtractor.from_pretrained(mdir)
    vocab = MultiLevelTokenizer(mdir).id_to_vocab
    levels = list(model.level_to_lm_head.keys())
    wavs = sorted(glob.glob(os.path.join(w, 'test-audio', '*.wav')))
    is_haraka = lambda t: unicodedata.category(t[0]) == 'Mn'
    groups = ['letters', 'harakat'] + [lv for lv in levels if lv != 'phonemes']

    def seqs(logits_by_level):
        out = {}
        for lv, lg in logits_by_level.items():
            ids = ctc_greedy(lg.argmax(-1), PAD_TOKEN_IDX)
            if lv == 'phonemes':
                toks = [vocab[lv][i] for i in ids]
                out['letters'] = [t for t in toks if not is_haraka(t)]
                out['harakat'] = [t for t in toks if is_haraka(t)]
            else:
                out[lv] = ids
        return out

    refs = []
    for p in wavs:
        x = sf.read(p, dtype='float32')[0]
        f = ext(x, sampling_rate=16000, return_tensors='pt')
        with torch.no_grad():
            lg = model(f['input_features'], attention_mask=f['attention_mask'], return_dict=False)[0]
        lg = {lv: lg[lv][0].numpy() for lv in levels}
        refs.append((p, x, lg, seqs(lg)))

    table = []
    for mp in a.models:
        sess = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
        names = [o.name for o in sess.get_outputs()]
        assert names == ['logits_' + lv for lv in levels], names
        err = {g: 0 for g in groups}
        tot = {g: 0 for g in groups}
        frames = {lv: [] for lv in levels}
        for p, x, ref_lg, ref_s in refs:
            outs = dict(zip(levels, (o[0] for o in sess.run(None, {'waveform': x[None]}))))
            s = seqs(outs)
            for g in groups:
                err[g] += edit_distance(ref_s[g], s[g])
                tot[g] += len(ref_s[g])
            for lv in levels:
                frames[lv].append(float((outs[lv].argmax(-1) == ref_lg[lv].argmax(-1)).mean()))
        size = os.path.getsize(mp) + (os.path.getsize(mp + '.data') if os.path.exists(mp + '.data') else 0)
        row = {'model': os.path.basename(mp), 'bytes': size,
               'match': {g: 1 - err[g] / max(1, tot[g]) for g in groups},
               'errors': err, 'refLength': tot,
               'framesMin': {lv: min(v) for lv, v in frames.items()}}
        table.append(row)
        print('\n%s (%.0f MB)' % (row['model'], size / 1e6))
        for g in groups:
            print('  %-20s %7.2f%%  (%d/%d)  frames min %.2f%%' % (
                g, 100 * row['match'][g], err[g], tot[g],
                100 * row['framesMin']['phonemes' if g in ('letters', 'harakat') else g]), flush=True)
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as fh:
            json.dump(table, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
