"""يجمع ملفات الإصدارين ويحسب الحجم و SHA-256، ويكتب assets.json.

    python scripts/make_assets.py --models <work-dir> \
        --data <data-dir>/_release

    python scripts/make_assets.py ... --verify     # ينزّل كل رابط ويتحقق من SHA-256
"""
import argparse, hashlib, json, os, sys, urllib.request

REPO = 'maher05f/zad-assets'
BASE = 'https://github.com/%s/releases/download' % REPO
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QP_VERSION = '2026-11-03'

MODELS_TAG, DATA_TAG = 'models-v1', 'data-v1'
MODEL_LICENSES = ['LICENSE-whisper-quran.txt', 'NOTICE-whisper-quran.txt',
                  'LICENSE-muaalem-QML-1.0.txt', 'NOTICE-muaalem.txt', 'LICENSE-phonetic-ref.txt']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def info(path, tag):
    name = os.path.basename(path)
    return {'url': '%s/%s/%s' % (BASE, tag, name), 'size': os.path.getsize(path), 'sha256': sha256(path)}


def release_files(models, data):
    """{tag: [مسار محلي]} — ما يُرفع في كل إصدار."""
    out = os.path.join(models, 'out')
    lic = os.path.join(models, 'licenses')
    m = [os.path.join(out, f) for f in ('ggml-quran-base-q5_0.bin', 'muaalem-int8.onnx', 'muaalem-fp32.onnx',
                                        'muaalem-levels.json', 'phonetic_ref.json')]
    # muaalem-fp32.onnx.data أكبر من 2GB فيُرفع أجزاءً (scripts/split_parts.py)
    m += [os.path.join(out, f) for f in sorted(os.listdir(out)) if f.startswith('muaalem-fp32.onnx.data.part')]
    m += [os.path.join(lic, f) for f in MODEL_LICENSES]
    d = sorted(os.path.join(data, f) for f in os.listdir(data) if f.endswith('.zip'))
    return {MODELS_TAG: m, DATA_TAG: d}


def build(models, data):
    files = release_files(models, data)
    for paths in files.values():
        for p in paths:
            if not os.path.isfile(p):
                sys.exit('ناقص: ' + p)
    byname = {os.path.basename(p): (p, tag) for tag, ps in files.items() for p in ps}
    f = lambda name: info(*byname[name])

    def extra(names):
        return [dict(file=n, **f(n)) for n in names]

    assets = [
        dict(id='model-light', type='model', version=1, **f('ggml-quran-base-q5_0.bin'), minRamGb=3,
             license=extra(['LICENSE-whisper-quran.txt', 'NOTICE-whisper-quran.txt'])),
        dict(id='model-strong', type='model', version=1, **f('muaalem-int8.onnx'), minRamGb=6,
             extra=extra(['muaalem-levels.json', 'phonetic_ref.json']),
             license=extra(['LICENSE-muaalem-QML-1.0.txt', 'NOTICE-muaalem.txt', 'LICENSE-phonetic-ref.txt'])),
    ]
    # model-full: أوزان fp32 في muaalem-fp32.onnx.data (تُدمج أجزاؤه بالترتيب)، والرسم muaalem-fp32.onnx
    # في extra ويجب أن يكون بجانب ملف الأوزان بهذا الاسم. levels و phonetic_ref نفس ملفات model-strong.
    data = os.path.join(models, 'out', 'muaalem-fp32.onnx.data')
    parts = sorted((p for p in byname if p.startswith('muaalem-fp32.onnx.data.part')), key=lambda s: int(s.rsplit('part', 1)[1]))
    assets.append(dict(
        id='model-full', type='model', version=1, minRamGb=12,
        file='muaalem-fp32.onnx.data', size=os.path.getsize(data), sha256=sha256(data),
        parts=[f(p) for p in parts],
        extra=extra(['muaalem-fp32.onnx', 'muaalem-levels.json', 'phonetic_ref.json']),
        license=extra(['LICENSE-muaalem-QML-1.0.txt', 'NOTICE-muaalem.txt', 'LICENSE-phonetic-ref.txt'])))
    assert sum(p['size'] for p in assets[-1]['parts']) == assets[-1]['size']
    for p in files[DATA_TAG]:
        pid = os.path.basename(p)[:-4]
        a = dict(id=pid, type='data', version=1, **info(p, DATA_TAG))
        if pid.startswith('quranpedia-'):
            a['sourceVersion'] = QP_VERSION
        assets.append(a)
    with open(os.path.join(HERE, 'assets.json'), 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(assets, fh, ensure_ascii=False, indent=1)
        fh.write('\n')
    return assets


def urls(assets):
    seen = set()
    for a in assets:
        items = a.get('parts') or [a]
        for e in items + a.get('extra', []) + a.get('license', []):
            if e['url'] not in seen:
                seen.add(e['url'])
                yield e['url'], e['size'], e['sha256']


def fetch(url, *hashers):
    n = 0
    with urllib.request.urlopen(url) as r:
        for chunk in iter(lambda: r.read(1 << 20), b''):
            for h in hashers:
                h.update(chunk)
            n += len(chunk)
    return n


def verify(assets):
    """ينزّل كل رابط مرة واحدة ويتحقق منه؛ والملف المقسوم يُتحقق منه أيضاً بعد الدمج."""
    bad, done = 0, set()
    for a in assets:
        whole = hashlib.sha256() if a.get('parts') else None
        for e in (a.get('parts') or [a]) + a.get('extra', []) + a.get('license', []):
            if e['url'] in done and not (whole and e in a['parts']):
                continue
            done.add(e['url'])
            h = hashlib.sha256()
            n = fetch(e['url'], h, *([whole] if whole and e in a['parts'] else []))
            ok = n == e['size'] and h.hexdigest() == e['sha256']
            bad += not ok
            print('%s %s' % ('✓' if ok else '✗', e['url']), flush=True)
        if whole:
            ok = whole.hexdigest() == a['sha256']
            bad += not ok
            print('%s %s بعد دمج %d أجزاء' % ('✓' if ok else '✗', a['file'], len(a['parts'])), flush=True)
    print('سليم: %d · تالف: %d' % (len(done) - bad, bad))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--list', action='store_true', help='اطبع الملفات لكل إصدار')
    a = ap.parse_args()
    if a.list:
        for tag, ps in release_files(a.models, a.data).items():
            print(tag, *ps, sep='\n  ')
        return
    assets = build(a.models, a.data)
    print('assets.json: %d عنصراً' % len(assets))
    if a.verify:
        sys.exit(1 if verify(assets) else 0)


if __name__ == '__main__':
    main()
