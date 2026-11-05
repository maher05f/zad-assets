"""يبني حزم بيانات زاد (zip) من مجلد XCZAD دون تعديل أي ملف.

    python scripts/build_packs.py --src <data-dir> [--out <dir>] [pack ...]

الحزم حتمية: ترتيب ثابت وتاريخ ثابت، فإعادة البناء من الملفات نفسها تعطي SHA-256 نفسه.
الملفات المضغوطة أصلاً (.gz و.zip) تُخزَّن دون ضغط، والباقي يُضغط deflate 9.
"""
import argparse, glob, hashlib, json, os, sys, zipfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIC = os.path.join(HERE, 'licenses')
STAMP = (2026, 11, 4, 0, 0, 0)
QP_VERSION = '2026-11-03'

MUSHAF_SOURCE = (
    '# mushaf-qpc2 (%s)\n\n'
    "- Fonts: King Fahd Glorious Qur'an Printing Complex (مجمع الملك فهد لطباعة المصحف الشريف), "
    'unmodified — https://qurancomplex.gov.sa\n'
    '- Layout: zonetecde/mushaf-layout (ISC) — https://github.com/zonetecde/mushaf-layout\n'
)
QP_SOURCE = 'Source: Quranpedia.net — https://quranpedia.net — data version %s\n' % QP_VERSION
CORPUS_SOURCE = (
    'Source: Quranic Arabic Corpus v0.4 by Dr. Kais Dukes, University of Leeds — https://corpus.quran.com\n'
)
TREEBANK_SOURCE = (
    "I'rab syntax: The Quranic Treebank — https://github.com/NoorBayan/Quranic — MIT "
    '(see LICENSE-MIT-QuranicTreebank.txt), built on the Quranic Arabic Corpus.\n'
)

PACKS = {
    'mushaf-qpc2-base': dict(
        files=['01-mushaf/fonts-qpc2-ttf/QCF2BSML.ttf', '01-mushaf/layout-qpc2/page-[0-9][0-9][0-9].json'],
        strip='01-mushaf/',
        license={'LICENSE': 'Mushaf-LICENSE.md'},
        source=MUSHAF_SOURCE % 'base: page layout for all 604 pages + basmala font QCF2BSML',
    ),
    'quranpedia-core': dict(
        files=['02-quranpedia/%s' % f for f in (
            'morphology.json.gz', 'meanings.json.gz', 'surahs.json.gz', 'surahs-index.json.gz', 'books.json.gz',
            'nasekh*.json.gz', 'mutshabeh.json.gz', 'similar.json.gz', 'topics*.json.gz')],
        strip='02-quranpedia/',
        license={'LICENSE': 'Quranpedia-LICENSE.md', 'LICENSE-GPL-3.0.txt': 'GPL-3.0.txt'},
        source='# quranpedia-core\n\n' + QP_SOURCE +
               '\nmorphology.json.gz contains word morphology from the Quranic Arabic Corpus '
               '(https://corpus.quran.com), GNU GPL v3 — see LICENSE-GPL-3.0.txt.\n',
    ),
    'quranpedia-e3rab': dict(
        files=['02-quranpedia/e3rab.json.gz', '02-quranpedia/e3rab_books-all.zip', '02-quranpedia/syntax.json.gz'],
        strip='02-quranpedia/',
        license={'LICENSE': 'Quranpedia-LICENSE.md', 'LICENSE-MIT-QuranicTreebank.txt': 'MIT-QuranicTreebank.txt'},
        source='# quranpedia-e3rab\n\n' + QP_SOURCE + '\n' + TREEBANK_SOURCE,
    ),
    'quranpedia-qiraat': dict(
        files=['02-quranpedia/qiraat.json.gz', '02-quranpedia/mushafs-all.zip'],
        strip='02-quranpedia/',
        license={'LICENSE': 'Quranpedia-LICENSE.md'},
        source='# quranpedia-qiraat\n\n' + QP_SOURCE,
    ),
    'quranpedia-asbab': dict(
        files=['02-quranpedia/asbab.json.gz', '02-quranpedia/asbab_books-all.zip'],
        strip='02-quranpedia/',
        license={'LICENSE': 'Quranpedia-LICENSE.md'},
        source='# quranpedia-asbab\n\n' + QP_SOURCE,
    ),
    'corpus-morphology': dict(
        files=['03-corpus-morphology/quran-morphology.txt', '03-corpus-morphology/morphology-terms-ar.json',
               '03-corpus-morphology/README.md'],
        strip='03-corpus-morphology/',
        license={'LICENSE': 'GPL-3.0.txt'},
        source='# corpus-morphology\n\n' + CORPUS_SOURCE +
               '\nThis is the modified fork https://github.com/mustafa0x/quran-morphology '
               '(the changes are listed in README.md). License: GNU GPL v3 — see LICENSE.\n'
               '\nRelated: ' + TREEBANK_SOURCE.replace(' (see LICENSE-MIT-QuranicTreebank.txt)', ''),
    ),
}


# بداية كل جزء في مصحف المدينة (604 صفحات)
JUZ_START = [1, 22, 42, 62, 82, 102, 122, 142, 162, 182, 201, 222, 242, 262, 282, 302, 322, 342, 362, 382,
             402, 422, 442, 462, 482, 502, 522, 542, 562, 582, 605]
for j in range(30):
    PACKS['mushaf-qpc2-juz%02d' % (j + 1)] = dict(
        files=['01-mushaf/fonts-qpc2-ttf/QCF2%03d.ttf' % p for p in range(JUZ_START[j], JUZ_START[j + 1])],
        strip='01-mushaf/',
        license={'LICENSE': 'Mushaf-LICENSE.md'},
        source=MUSHAF_SOURCE % ('juz %d: page fonts %d-%d' % (j + 1, JUZ_START[j], JUZ_START[j + 1] - 1)),
    )
PACKS['font-uthmanic-hafs'] = dict(
    files=['01-mushaf/fonts-other/UthmanicHafs_V22/uthmanic_hafs_v22.ttf',
           '01-mushaf/fonts-other/UthmanicHafs_V22/UthmanicHafs v22.json'],
    strip='01-mushaf/fonts-other/UthmanicHafs_V22/',
    license={'LICENSE': 'KFGQPC-Font-EULA.txt'},
    source='# font-uthmanic-hafs\n\n'
           "- KFGQPC HAFS Uthmanic Script v22 + matching text, King Fahd Glorious Qur'an Printing Complex "
           '(مجمع الملك فهد لطباعة المصحف الشريف), unmodified — http://fonts.qurancomplex.gov.sa/\n'
           '- License: the EULA embedded in the font (see LICENSE). Free to use, copy and distribute; '
           'must not be sold or modified.\n',
)


def entry(name, data, compress):
    zi = zipfile.ZipInfo(name, STAMP)
    zi.compress_type = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    zi.external_attr = 0o644 << 16
    return zi, data


def build(src, out, name, spec):
    paths = []
    for pat in spec['files']:
        hits = sorted(glob.glob(os.path.join(src, pat)))
        if not hits:
            sys.exit('%s: لا ملفات تطابق %s' % (name, pat))
        paths += hits
    items = []
    for p in paths:
        rel = os.path.relpath(p, src).replace('\\', '/')
        rel = rel[len(spec['strip']):] if rel.startswith(spec['strip']) else rel
        items.append(entry(rel, open(p, 'rb').read(), not p.endswith(('.gz', '.zip'))))
    for arc, lic in spec['license'].items():
        items.append(entry(arc, open(os.path.join(LIC, lic), 'rb').read(), True))
    items.append(entry('SOURCE.md', spec['source'].encode('utf-8'), True))
    zpath = os.path.join(out, name + '.zip')
    with zipfile.ZipFile(zpath, 'w', compresslevel=9) as z:
        for zi, data in items:
            z.writestr(zi, data)
    raw = sum(len(d) for _, d in items)
    data = open(zpath, 'rb').read()
    return dict(id=name, file=name + '.zip', size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                files=len(items), rawSize=raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out')
    ap.add_argument('packs', nargs='*')
    a = ap.parse_args()
    out = a.out or os.path.join(a.src, '_release')
    os.makedirs(out, exist_ok=True)
    res = [build(a.src, out, n, PACKS[n]) for n in (a.packs or PACKS)]
    for r in res:
        print('%-18s %9.1f MB  (قبل الضغط %6.1f MB، %d ملفاً)' % (r['id'], r['size'] / 1e6, r['rawSize'] / 1e6, r['files']))
    with open(os.path.join(out, 'packs.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
