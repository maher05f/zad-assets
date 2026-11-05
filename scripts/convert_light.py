"""النسخة الخفيفة: tarteel-ai/whisper-base-ar-quran → GGML q5_0 لـ whisper.cpp.

    python scripts/convert_light.py --work <work-dir>

يفترض داخل --work:
  src/whisper.cpp            (v1.9.5)  git clone https://github.com/ggml-org/whisper.cpp
  src/whisper                          git clone https://github.com/openai/whisper
  src/whisper-base-ar-quran            git clone https://huggingface.co/tarteel-ai/whisper-base-ar-quran
  bin/whisper/Release        ملفات whisper.cpp الرسمية (whisper-bin-x64.zip من الإصدار b5454 = v1.9.5)
  .venv                      Python فيه torch (CPU) و transformers
والناتج: out/ggml-quran-base-q5_0.bin
"""
import argparse, json, os, subprocess, sys


def run(cmd, **kw):
    print('>', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--wav', nargs='*', default=[])
    a = ap.parse_args()
    w = a.work
    py = os.path.join(w, '.venv', 'Scripts', 'python.exe')
    binr = os.path.join(w, 'bin', 'whisper', 'Release')
    out = os.path.join(w, 'out')
    tmp = os.path.join(w, 'tmp', 'ggml-f16')
    os.makedirs(out, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)

    # convert-h5-to-ggml يكتب max_length (إعداد التوليد = 1024 هنا) على أنه n_text_ctx،
    # والصحيح max_target_positions (448). نجهّز نسخة من مجلد النموذج فيها max_length=448
    # (الأوزان رابط صلب، والأصل لا يُمس).
    src_model = os.path.join(w, 'src', 'whisper-base-ar-quran')
    staged = os.path.join(w, 'tmp', 'whisper-base-ar-quran')
    os.makedirs(staged, exist_ok=True)
    for name in os.listdir(src_model):
        s, d = os.path.join(src_model, name), os.path.join(staged, name)
        if not os.path.isfile(s) or name == 'config.json':
            continue
        if os.path.exists(d):
            os.remove(d)
        os.link(s, d)
    with open(os.path.join(src_model, 'config.json'), encoding='utf-8') as fh:
        cfg = json.load(fh)
    cfg['max_length'] = cfg['max_target_positions']
    with open(os.path.join(staged, 'config.json'), 'w', encoding='utf-8') as fh:
        json.dump(cfg, fh, indent=2)

    f16 = os.path.join(tmp, 'ggml-model.bin')
    run([py, '-I', os.path.join(w, 'src', 'whisper.cpp', 'models', 'convert-h5-to-ggml.py'),
         staged, os.path.join(w, 'src', 'whisper'), tmp])
    q5 = os.path.join(out, 'ggml-quran-base-q5_0.bin')
    run([os.path.join(binr, 'whisper-quantize.exe'), f16, q5, 'q5_0'])
    for wav in a.wav:
        run([os.path.join(binr, 'whisper-cli.exe'), '-m', q5, '-l', 'ar', '-nt', '-f', wav])
    print('ok:', q5, os.path.getsize(q5))


if __name__ == '__main__':
    sys.exit(main())
