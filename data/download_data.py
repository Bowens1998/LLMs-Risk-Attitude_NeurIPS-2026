"""Download the public dataset from Hugging Face into data/hf/ (standard library only).

Usage:
    python data/download_data.py              # all files
    python data/download_data.py --skip-meta  # skip the large per-run *_meta.json provenance files
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from riskattitude.paths import DATA_DIR, HF_REPO  # noqa: E402

API = f'https://huggingface.co/api/datasets/{HF_REPO}/tree/main?recursive=true'
RESOLVE = f'https://huggingface.co/datasets/{HF_REPO}/resolve/main/'


def list_files():
    with urllib.request.urlopen(API, timeout=120) as r:
        return [(e['path'], e.get('size', 0)) for e in json.load(r) if e['type'] == 'file']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-meta', action='store_true', help='skip *_meta.json provenance files')
    ap.add_argument('--force', action='store_true', help='re-download files that already exist')
    args = ap.parse_args()

    files = [(p, s) for p, s in list_files() if not p.startswith('.')]
    if args.skip_meta:
        files = [(p, s) for p, s in files if not p.endswith('_meta.json')]
    for i, (path, size) in enumerate(files, 1):
        dest = DATA_DIR / path
        if dest.exists() and dest.stat().st_size == size and not args.force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f'[{i}/{len(files)}] {path} ({size / 1e6:.1f} MB)')
        url = RESOLVE + urllib.request.quote(path)
        with urllib.request.urlopen(url, timeout=600) as r, open(dest, 'wb') as f:
            f.write(r.read())
    print(f'Dataset ready in {DATA_DIR}')


if __name__ == '__main__':
    main()
