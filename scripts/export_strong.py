"""النسخة القوية: obadx/muaalem-model-v3_2 → ONNX (opset 17) → int8.

    <venv>/python scripts/export_strong.py --work <work-dir>

يفترض داخل --work:
  src/muaalem-model-v3_2   git clone https://huggingface.co/obadx/muaalem-model-v3_2
  src/quran-muaalem        git clone https://github.com/obadx/quran-muaalem   (لصنف النموذج والمُرمِّز)
  test-audio/*.wav         16kHz قناة واحدة
والناتج في out/:
  muaalem-int8.onnx        يقبل الصوت الخام: waveform float32 [1, N] بين -1 و1، 16kHz
                           (حساب fbank: 80 مِل، تطبيع لكل مِل، تكديس 2 → 160 داخل النموذج)
  muaalem-levels.json      ترتيب المستويات ومفرداتها
  muaalem-test/            ملف اختبار لـ Kotlin: صوت + ميزات متوقعة + مخرجات متوقعة
"""
import argparse, glob, json, os, sys, time

import numpy as np
import soundfile as sf
import torch
from torch import nn

SR = 16000


def kaldi_fbank_consts(extractor):
    """ثوابت fbank نفسها التي يستعملها SeamlessM4TFeatureExtractor."""
    n = np.arange(400)
    k = np.arange(257)
    ang = 2 * np.pi * np.outer(n, k) / 512  # rfft بطول 512 لإطار 400 (الباقي أصفار)
    return dict(
        window=torch.tensor(extractor.window, dtype=torch.float32),
        cos=torch.tensor(np.cos(ang), dtype=torch.float32),
        sin=torch.tensor(-np.sin(ang), dtype=torch.float32),
        mel=torch.tensor(extractor.mel_filters, dtype=torch.float32),  # [257, 80]
    )


class Fbank(nn.Module):
    """نسخة torch من SeamlessM4TFeatureExtractor لدفعة واحدة، قابلة للتصدير إلى ONNX.

    تُرجع (input_features [1, T2, 160], attention_mask [1, T2]) كما يُرجعها المستخرج الأصلي
    مع padding=True و pad_to_multiple_of=2 و padding_value=1.
    """

    def __init__(self, consts, padding_value=1.0):
        super().__init__()
        for k, v in consts.items():
            self.register_buffer(k, v)
        self.padding_value = padding_value

    def forward(self, waveform):
        x = waveform[0] * 32768.0
        n_frames = (x.shape[0] - 400) // 160 + 1
        idx = torch.arange(n_frames).unsqueeze(1) * 160 + torch.arange(400).unsqueeze(0)
        fr = x[idx]  # [T, 400]
        fr = fr - fr.mean(dim=1, keepdim=True)
        fr = torch.cat([fr[:, :1] * (1 - 0.97), fr[:, 1:] - 0.97 * fr[:, :-1]], dim=1)
        fr = fr * self.window
        re, im = fr @ self.cos, fr @ self.sin
        power = re * re + im * im  # [T, 257]
        feat = torch.log(torch.clamp(power @ self.mel, min=1.192092955078125e-07))  # [T, 80]
        mean = feat.mean(dim=0, keepdim=True)
        var = ((feat - mean) ** 2).sum(dim=0, keepdim=True) / (feat.shape[0] - 1)
        feat = (feat - mean) / torch.sqrt(var + 1e-7)
        # إكمال عدد الإطارات إلى زوجي بإطار قيمته 1 (padding_value) وقناعه 0
        odd = feat.shape[0] % 2
        pad = torch.full((odd, 80), self.padding_value)
        feat = torch.cat([feat, pad], dim=0)
        mask = torch.cat([torch.ones(feat.shape[0] - odd, dtype=torch.long), torch.zeros(odd, dtype=torch.long)])
        t2 = feat.shape[0] // 2
        return feat.reshape(1, t2, 160), mask.reshape(t2, 2)[:, 1].reshape(1, t2)


class PointwiseAsMatMul(nn.Module):
    """Conv1d بنواة 1 مكتوبة ضرب مصفوفات، ليُكمَّم وزنها مع MatMul (النتيجة نفسها)."""

    def __init__(self, conv):
        super().__init__()
        assert conv.kernel_size == (1,) and conv.groups == 1 and conv.stride == (1,)
        self.weight_t = nn.Parameter(conv.weight.detach()[:, :, 0].t().contiguous())  # [in, out]
        self.bias = None if conv.bias is None else nn.Parameter(conv.bias.detach())

    def forward(self, x):  # [B, C, T]
        y = x.transpose(1, 2) @ self.weight_t
        if self.bias is not None:
            y = y + self.bias
        return y.transpose(1, 2)


def pointwise_to_matmul(model):
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Conv1d) and child.kernel_size == (1,) and child.groups == 1:
                setattr(mod, cname, PointwiseAsMatMul(child))
                n += 1
    return n


class RawMuaalem(nn.Module):
    def __init__(self, model, fbank, levels):
        super().__init__()
        self.model, self.fbank, self.levels = model, fbank, levels

    def forward(self, waveform):
        feats, mask = self.fbank(waveform)
        logits = self.model(feats, attention_mask=mask, return_dict=False)[0]
        return tuple(logits[lv] for lv in self.levels)


def ctc_greedy(ids, blank):
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            out.append(int(i))
        prev = i
    return out


def edit_distance(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


def save_full(fp32, dst):
    """model-full: نسخة fp32 كما صُدِّرت، بلا أي ضغط، في ملفين: الرسم dst وأوزانه dst + '.data'
    (ملف ONNX واحد لا يتجاوز 2GB)."""
    import onnx
    m = onnx.load(fp32)
    for p in (dst, dst + '.data'):
        if os.path.exists(p):
            os.remove(p)
    onnx.save_model(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=os.path.basename(dst) + '.data', size_threshold=1024)
    print('fp32: %s %d + %d bytes' % (os.path.basename(dst), os.path.getsize(dst), os.path.getsize(dst + '.data')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--skip-export', action='store_true')
    a = ap.parse_args()
    w = a.work
    sys.path.insert(0, os.path.join(w, 'src', 'quran-muaalem', 'src'))
    from transformers import AutoFeatureExtractor
    from quran_muaalem.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
    from quran_muaalem.modeling.multi_level_tokenizer import MultiLevelTokenizer
    from quran_muaalem.modeling.vocab import PAD_TOKEN_IDX

    mdir = os.path.join(w, 'src', 'muaalem-model-v3_2')
    out = os.path.join(w, 'out')
    tmp = os.path.join(w, 'tmp', 'onnx-fp32')
    os.makedirs(out, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)

    torch.manual_seed(0)
    model = Wav2Vec2BertForMultilevelCTC.from_pretrained(mdir).eval().float()
    extractor = AutoFeatureExtractor.from_pretrained(mdir)
    tok = MultiLevelTokenizer(mdir)
    levels = list(model.level_to_lm_head.keys())
    id_to_vocab = tok.id_to_vocab
    with open(os.path.join(out, 'muaalem-levels.json'), 'w', encoding='utf-8') as fh:
        json.dump({
            'model': 'obadx/muaalem-model-v3_2',
            'input': {'name': 'waveform', 'shape': [1, 'N'], 'dtype': 'float32', 'sampleRate': SR,
                      'range': '[-1, 1]', 'note': 'fbank (80 mel, per-mel normalization, stride 2 → 160) is inside the model'},
            'blankId': PAD_TOKEN_IDX,
            'outputs': [{'name': 'logits_' + lv, 'level': lv, 'vocabSize': model.config.level_to_vocab_size[lv],
                         'idToToken': {str(k): v for k, v in sorted(id_to_vocab[lv].items())}}
                        for lv in levels],
            'sifatEnglish': {lv: {str(k): v for k, v in m.items()} for lv, m in tok.sifat_to_en_vocab.items()},
        }, fh, ensure_ascii=False, indent=1)
    print('levels:', levels)

    fbank = Fbank(kaldi_fbank_consts(extractor), float(extractor.padding_value)).eval()
    import copy
    export_model = copy.deepcopy(model)
    print('pointwise conv → matmul:', pointwise_to_matmul(export_model))
    raw = RawMuaalem(export_model, fbank, levels).eval()
    wavs = sorted(glob.glob(os.path.join(w, 'test-audio', '*.wav')))

    # 1) fbank داخل torch مقابل المستخرج الأصلي
    for p in wavs:
        x, sr = sf.read(p, dtype='float32')
        assert sr == SR and x.ndim == 1, p
        ref = extractor(x, sampling_rate=SR, return_tensors='pt')
        with torch.no_grad():
            f, m = fbank(torch.from_numpy(x)[None])
        assert f.shape == ref['input_features'].shape and torch.equal(m, ref['attention_mask'].long()), p
        print('fbank %-24s max|Δ| = %.2e' % (os.path.basename(p), (f - ref['input_features']).abs().max()))

    fp32 = os.path.join(tmp, 'muaalem-fp32.onnx')
    q8 = os.path.join(out, 'muaalem-int8.onnx')
    if not a.skip_export:
        import shutil
        shutil.rmtree(tmp)
        os.makedirs(tmp)
        t0 = time.time()
        dummy = torch.from_numpy(sf.read(wavs[0], dtype='float32')[0])[None]
        torch.onnx.export(
            raw, (dummy,), fp32, dynamo=False, opset_version=17,
            input_names=['waveform'], output_names=['logits_' + lv for lv in levels],
            dynamic_axes={'waveform': {1: 'samples'}, **{'logits_' + lv: {1: 'frames'} for lv in levels}},
        )
        print('fp32 export %.0fs' % (time.time() - t0))
        save_full(fp32, os.path.join(out, 'muaalem-fp32.onnx'))
        # int8 للأوزان فقط (MatMulNBits، كتل 128، متماثل)؛ الحساب يبقى float32.
        # التكميم الديناميكي المعتاد (أوزان + تنشيطات int8) أنزل تطابق الحروف إلى ~89%،
        # وهذا يطابق PyTorch تماماً على ملفات الاختبار. Conv المتبقية (المحوّل) تبقى float32.
        import onnx
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig
        t0 = time.time()
        cfg = DefaultWeightOnlyQuantConfig(block_size=128, is_symmetric=True, bits=8, op_types_to_quantize=('MatMul',))
        q = MatMulNBitsQuantizer(onnx.load(fp32), bits=8, block_size=128, is_symmetric=True, algo_config=cfg,
                                 op_types_to_quantize=('MatMul',))
        q.process()
        onnx.save_model(q.model.model, q8)
        print('int8 %.0fs, %d bytes' % (time.time() - t0, os.path.getsize(q8)))

    # 2) مقارنة PyTorch (fp32 بالمستخرج الأصلي) مع ONNX int8 (صوت خام)
    import onnxruntime as ort
    sess = ort.InferenceSession(q8, providers=['CPUExecutionProvider'])
    report, tot_err, tot_len = [], 0, 0
    for p in wavs:
        x = sf.read(p, dtype='float32')[0]
        ref = extractor(x, sampling_rate=SR, return_tensors='pt')
        with torch.no_grad():
            pt = model(ref['input_features'], attention_mask=ref['attention_mask'], return_dict=False)[0]
        ox = sess.run(None, {'waveform': x[None]})
        row = {'file': os.path.basename(p)}
        for lv, o in zip(levels, ox):
            a_ids = pt[lv][0].argmax(-1).numpy()
            b_ids = o[0].argmax(-1)
            row['frames_' + lv] = float((a_ids == b_ids).mean())
            if lv == 'phonemes':
                ta = ''.join(id_to_vocab[lv][i] for i in ctc_greedy(a_ids, PAD_TOKEN_IDX))
                tb = ''.join(id_to_vocab[lv][i] for i in ctc_greedy(b_ids, PAD_TOKEN_IDX))
                e = edit_distance(ta, tb)
                tot_err, tot_len = tot_err + e, tot_len + len(ta)
                row.update(pytorch=ta, onnx=tb, char_match=1 - e / max(1, len(ta)))
        report.append(row)
        print('%-24s حروف %.2f%%  إطارات(phonemes) %.2f%%' % (row['file'], 100 * row['char_match'], 100 * row['frames_phonemes']))
    print('المجموع: تطابق الحروف %.2f%% (%d خطأ من %d)' % (100 * (1 - tot_err / tot_len), tot_err, tot_len))
    with open(os.path.join(out, 'muaalem-compare.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=1)

    # 3) ملف اختبار لـ Kotlin: أول ملف صوتي + الميزات + المخرجات المتوقعة
    td = os.path.join(out, 'muaalem-test')
    os.makedirs(td, exist_ok=True)
    x = sf.read(wavs[0], dtype='float32')[0]
    ref = extractor(x, sampling_rate=SR, return_tensors='np')
    ox = sess.run(None, {'waveform': x[None]})
    np.save(os.path.join(td, 'waveform.npy'), x)
    np.save(os.path.join(td, 'features.npy'), ref['input_features'][0].astype(np.float32))
    for lv, o in zip(levels, ox):
        np.save(os.path.join(td, 'logits_%s.npy' % lv), o[0].astype(np.float32))
    ids = ctc_greedy(ox[levels.index('phonemes')][0].argmax(-1), PAD_TOKEN_IDX)
    import shutil
    shutil.copyfile(wavs[0], os.path.join(td, os.path.basename(wavs[0])))
    with open(os.path.join(td, 'expected.json'), 'w', encoding='utf-8') as fh:
        json.dump({'audio': os.path.basename(wavs[0]), 'samples': int(x.shape[0]),
                   'featuresShape': list(ref['input_features'][0].shape),
                   'phonemeIds': ids, 'phonemes': ''.join(id_to_vocab['phonemes'][i] for i in ids)},
                  fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
