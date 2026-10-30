"""Prepare official OPUS-100 en-sk splits; never mix train/dev/test.

    python scripts/prepare_opus.py --train-size 200000 --seed 42
    python scripts/prepare_opus.py --root /tmp/testdir --archive /tmp/fixture.tar.gz
"""
import argparse
import hashlib
import json
import random
import tarfile
import urllib.request
from pathlib import Path

URL = 'https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-en-sk-v1.0.tar.gz'
NAME = 'opus-100-corpus-en-sk-v1.0.tar.gz'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    if dest.exists():
        print(f'REUSE archive: {dest} ({dest.stat().st_size:,} bytes)', flush=True)
        return
    tmp = dest.with_name(dest.name + '.partial')
    request = urllib.request.Request(url, headers={'User-Agent': 'ATMT-Assignment3/1.0'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open('wb') as out:
            while True:
                buf = response.read(1024 * 1024)
                if not buf:
                    break
                out.write(buf)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def prepare(a):
    root = Path(a.root).expanduser().resolve()
    out = root / 'data' / 'opus100_en_sk'
    out.mkdir(parents=True, exist_ok=True)
    archive = Path(a.archive).expanduser().resolve() if a.archive else out / NAME
    if a.archive:
        if not archive.is_file():
            raise FileNotFoundError(f'Archive not found: {archive}')
    else:
        download(URL, archive)

    # Validate the whole gzip/tar as we stream its members.
    expected = {(split, lang): f'opus.en-sk-{split}.{lang}'
                for split in ('train', 'dev', 'test') for lang in ('sk', 'en')}
    with tarfile.open(archive, 'r:gz') as tar:
        members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
        missing = sorted(set(expected.values()) - set(members))
        if missing:
            raise RuntimeError(f'Archive is missing official files: {missing}; got: {sorted(members)[:20]}')
        counts = {}
        for split in ('train', 'dev', 'test'):
            raw = {}
            for lang in ('sk', 'en'):
                item = tar.extractfile(members[expected[split, lang]])
                if item is None:
                    raise RuntimeError(f'Cannot read {expected[split, lang]}')
                raw[lang] = item.read().decode('utf-8-sig').splitlines()
            if len(raw['sk']) != len(raw['en']):
                raise ValueError(f'{split}: raw sentence counts differ: sk={len(raw["sk"])} en={len(raw["en"])}')
            pairs = [(s.strip(), t.strip()) for s, t in zip(raw['sk'], raw['en']) if s.strip() and t.strip()]
            usable = len(pairs)
            if split == 'train':
                random.Random(a.seed).shuffle(pairs)
                pairs = pairs[:a.train_size]
            elif len(pairs) > 2000:
                pairs = pairs[:2000]
            for lang, col in (('sk', 0), ('en', 1)):
                target = out / f'{split}.{lang}'
                with target.open('w', encoding='utf-8') as f:
                    for row in pairs:
                        f.write(row[col] + '\n')
            counts[split] = {'original_pairs': len(raw['sk']), 'nonempty_pairs': usable, 'selected_pairs': len(pairs)}
            print(f'{split}: original={len(raw["sk"])} nonempty={usable} selected={len(pairs)}', flush=True)

    # Exact-pair overlap audit; do not change the official splits silently.
    sets = {}
    for split in ('train', 'dev', 'test'):
        sk = (out / f'{split}.sk').read_text(encoding='utf-8').splitlines()
        en = (out / f'{split}.en').read_text(encoding='utf-8').splitlines()
        assert len(sk) == len(en) == counts[split]['selected_pairs']
        sets[split] = set(zip(sk, en))
    overlap = {f'{a}_{b}': len(sets[a] & sets[b]) for a, b in (('train', 'dev'), ('train', 'test'), ('dev', 'test'))}
    print('Exact sentence-pair overlaps:', overlap, flush=True)

    meta = {
        'source_url': URL if not a.archive else str(archive),
        'archive_sha256': sha256(archive),
        'archive_size_bytes': archive.stat().st_size,
        'seed': a.seed, 'train_size_requested': a.train_size,
        'counts': counts, 'pair_overlap': overlap,
        'split_definition': 'official OPUS-100 train/dev/test; shuffled train subset only; discard pairs with either side blank',
        'files_sha256': {f'{s}.{l}': sha256(out / f'{s}.{l}') for s in ('train', 'dev', 'test') for l in ('sk','en')},
    }
    (out / 'manifest.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('PREPARE_OK', out / 'manifest.json', flush=True)
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=str(Path.home() / 'projects' / 'atmt_a3'))
    p.add_argument('--archive', default=None, help='Use an already downloaded archive, or a synthetic fixture')
    p.add_argument('--train-size', type=int, default=200000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.train_size < 1:
        p.error('--train-size must be >= 1')
    prepare(a)
