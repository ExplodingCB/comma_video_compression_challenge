#!/usr/bin/env python3
"""Inflate an attributed PR135 derivative with a lossless model container."""
import argparse
import json
from pathlib import Path
import zipfile

from runtime.f26_inflate import inflate_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir', type=Path)
    parser.add_argument('base')
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.base != '0':
        raise ValueError('This decoder supports the public challenge video 0')
    here = Path(__file__).resolve().parent
    archive_path = here / 'archive.zip'
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ['p']:
            raise ValueError('Expected one archive member named p')
        if (args.data_dir / 'p').read_bytes() != archive.read('p'):
            raise ValueError('Extracted payload differs from archive.zip')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    report = inflate_archive(archive_path, args.destination, renderer_dir=here / 'cpr1')
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
