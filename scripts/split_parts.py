"""يقسم ملفاً أكبر من 1.9 GiB إلى أجزاء متساوية تقريباً (حد GitHub Releases ملف 2 GiB).

    python scripts/split_parts.py out/muaalem-fp32.onnx.data          # يكتب .part1 .part2 … بجانبه
    python scripts/split_parts.py --merge out/muaalem-fp32.onnx.data tmp/merge/muaalem-fp32.onnx.data

يطبع الحجم و SHA-256 لكل جزء وللملف الكامل.
"""
import argparse, hashlib, json, math, os, sys

LIMIT = int(1.9 * 2**30)
CHUNK = 1 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for c in iter(lambda: fh.read(CHUNK), b''):
            h.update(c)
    return h.hexdigest()


def parts_of(path):
    out, i = [], 1
    while os.path.exists('%s.part%d' % (path, i)):
        out.append('%s.part%d' % (path, i))
        i += 1
    return out


def split(path):
    size = os.path.getsize(path)
    n = math.ceil(size / LIMIT)
    if n < 2:
        print('لا حاجة للتقسيم:', path)
        return []
    for old in parts_of(path):
        os.remove(old)
    per = math.ceil(size / n)
    out = []
    with open(path, 'rb') as src:
        for i in range(1, n + 1):
            p = '%s.part%d' % (path, i)
            left = min(per, size - (i - 1) * per)
            with open(p, 'wb') as dst:
                while left:
                    c = src.read(min(CHUNK, left))
                    dst.write(c)
                    left -= len(c)
            out.append(p)
    return out


def merge(path, dst):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, 'wb') as out:
        for p in parts_of(path):
            with open(p, 'rb') as fh:
                for c in iter(lambda: fh.read(CHUNK), b''):
                    out.write(c)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merge', nargs=2, metavar=('SRC', 'DST'))
    ap.add_argument('path', nargs='?')
    a = ap.parse_args()
    if a.merge:
        dst = merge(*a.merge)
        print(json.dumps({'file': dst, 'size': os.path.getsize(dst), 'sha256': sha256(dst)}))
        return
    parts = split(a.path)
    info = {'file': os.path.basename(a.path), 'size': os.path.getsize(a.path), 'sha256': sha256(a.path),
            'parts': [{'file': os.path.basename(p), 'size': os.path.getsize(p), 'sha256': sha256(p)} for p in parts]}
    assert sum(p['size'] for p in info['parts']) == info['size']
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    sys.exit(main())
