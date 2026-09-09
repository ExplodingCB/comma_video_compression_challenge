"""Copy the existing decoder beside a rebuilt archive in a new evaluation directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

SUBMISSION = Path(__file__).resolve().parents[1]


def prepare(archive_file, output):
    output = output.resolve()
    if output.exists():
        raise ValueError('Inflation directory must be new; existing raw results are never reused')
    with zipfile.ZipFile(archive_file) as archive:
        if archive.namelist() != ['p']:
            raise ValueError('Expected one member named p')
        payload = archive.read('p')
    output.mkdir(parents=True)
    shutil.copyfile(archive_file, output / 'archive.zip')
    for top in ('runtime', 'cpr1'):
        shutil.copytree(SUBMISSION / top, output / top,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.so'))
    for name in ('inflate.py', 'inflate.sh', 'prepare_dependencies.sh', 'LICENSE'):
        shutil.copyfile(SUBMISSION / name, output / name)
    (output / 'archive').mkdir()
    (output / 'archive/p').write_bytes(payload)
    print(json.dumps(dict(output=str(output), archive_sha256=hashlib.sha256((output / 'archive.zip').read_bytes()).hexdigest())))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive_file, args.output_dir)
