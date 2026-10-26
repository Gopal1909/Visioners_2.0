"""Dataset-free, standard-library verification of the published evidence files."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'docs/publication_manifest.json').read_text(encoding='utf-8'))
    failures=[];total=0
    for name,expected in manifest['files'].items():
        relative=Path(name);path=(root/relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root) or relative.parts[0].lower() in ('dataset','.git','.publish'):
            failures.append((name,'invalid publication path'));continue
        if not path.is_file():failures.append((name,'missing'));continue
        size=path.stat().st_size;total+=size
        if size!=expected['bytes'] or digest(path)!=expected['sha256']:failures.append((name,'bytes/hash mismatch'))
    if len(manifest['files'])!=manifest['file_count'] or total!=manifest['total_bytes']:failures.append(('manifest','totals mismatch'))
    if failures:
        print(json.dumps(dict(passed=False,failures=failures),indent=2));raise SystemExit(1)
    print(json.dumps(dict(passed=True,verified_files=manifest['file_count'],verified_bytes=total,
        original_dataset_required=False,scope='Uploaded file integrity; not regeneration of omitted research caches'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();verify(args.root)
