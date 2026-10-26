"""Copy a dataset-free evidence snapshot without touching original research files."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
GENERATED={'docs/publication_manifest.json','docs/publication_exclusions.json','docs/publication_security.json'}
PRUNE={'.git','__pycache__','.venv','venv','.pytest_cache','.mypy_cache','.ruff_cache','.idea','.vscode'}
ARRAY_DIRECTORIES={'assembled','features','cache','disk_support'}
SOURCE_SNAPSHOTS={'source','sources','final_sources','final_source_snapshot','pre_memory_recovery_sources',
                  'superseded_sources','pre_functional_optimizer_sources'}
MAX_FILE_BYTES=2*1024**2
TEXT_EXTENSIONS={'.py','.md','.txt','.json','.toml','.ini','.cfg','.yaml','.yml','.js','.mjs','.ts','.html','.log','.csv'}
SECRET_PATTERNS={
    'github_token':re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})'),
    'openai_token':re.compile(rb'sk-(?:proj-|svcacct-)[A-Za-z0-9_-]{40,}'),
    'aws_access_key':re.compile(rb'AKIA[0-9A-Z]{16}'),
    'private_key':re.compile(rb'-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----'),
    'credential_url':re.compile(rb'https?://[^\s/:@]+:[^\s/@]{4,}@'),
}


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def selected_files():
    """Validation-selected V2 weights and the small historical demo closure."""
    keep=set();base=ROOT/'results/learning_v2_coverage_2026-10-25_v1'
    for fold in range(5):
        folder=base/f'fold_{fold}';selection=json.loads((folder/'learning/selection.json').read_text())
        names=[selection['primary_variant']]
        if fold==0:names.append(selection['secondary_robust_variant'])
        for name in names:keep.add((folder/f'learning/{name}_seed42/best.pt').relative_to(ROOT).as_posix())
    bundle=ROOT/'final_submission/artifacts'
    for name in json.loads((bundle/'checksums.json').read_text()):keep.add((bundle/name).relative_to(ROOT).as_posix())
    return keep


def excluded_reason(relative,keep=None):
    parts=relative.parts;low=tuple(p.lower() for p in parts);name=relative.name.lower();suffix=relative.suffix.lower()
    if low[0]=='dataset':return 'original_dataset'
    if low[0]=='.publish' or low[0].startswith('.chart-data-'):return 'local_staging_or_ui'
    if any(p in PRUNE for p in low):return 'environment_or_bytecode'
    if name.startswith('.env') and name!='.env.example':return 'credential_configuration'
    if suffix in ('.pem','.key') or name.startswith(('id_rsa','id_ed25519')):return 'credential_configuration'
    if suffix in ('.pyc','.pyo'):return 'bytecode'
    if relative.as_posix() in GENERATED:return 'generated_publication_metadata'
    if keep and relative.as_posix() in keep:return None
    if suffix in ('.npy','.npz','.zip','.pt','.joblib'):return 'large_or_redundant_binary_artifact'
    if any(p in SOURCE_SNAPSHOTS for p in low) and suffix!='.json':return 'duplicate_archived_source_copy'
    if suffix in ('.npy','.npz') and any(p in ARRAY_DIRECTORIES for p in low):return 'regenerable_feature_or_support_array'
    if suffix=='.npz' and ('features' in name or 'patches' in name):return 'regenerable_feature_or_patch_array'
    if low[0]=='results' and 'epochs' in low and suffix=='.npz' and name.startswith('pair_'):return 'regenerable_fresh_epoch_array'
    if low[0]=='results' and len(low)>1 and low[1].startswith(('demo_local','publication_validation')):return 'local_publication_validation_output'
    return None


def scan_credentials(path):
    if path.suffix.lower() not in TEXT_EXTENSIONS and not path.name.startswith('.'):return []
    findings=set();tail=b''
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(131072),b''):
            data=tail+block
            for label,pattern in SECRET_PATTERNS.items():
                if pattern.search(data):findings.add(label)
            tail=data[-512:]
    return sorted(findings)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def copy_exact(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    sha=digest(source)
    if destination.exists():
        if digest(destination)!=sha:raise RuntimeError(f'Conflicting destination; original preserved: {destination}')
    else:shutil.copy2(source,destination)
    assert digest(destination)==sha
    return dict(bytes=source.stat().st_size,sha256=sha)


def export(destination):
    destination=Path(destination).resolve()
    intended=(ROOT/'.publish').resolve()
    if not destination.is_relative_to(intended) or destination==intended:
        raise ValueError('Destination must be a dedicated checkout beneath this workspace .publish directory')
    if not (destination/'.git').exists():raise ValueError('Clone the intended repository into destination first')
    included=[];excluded=[];directories=[];findings=[];keep=selected_files()
    print('Inventory and credential-pattern scan started',flush=True)
    for current,children,filenames in os.walk(ROOT):
        relative_dir=Path(current).relative_to(ROOT)
        for child in list(children):
            relative=relative_dir/child
            if child.lower() in PRUNE or relative.parts[0].lower() in ('dataset','.publish') or relative.parts[0].startswith('.chart-data-'):
                children.remove(child);directories.append(dict(path=relative.as_posix(),reason='dataset_environment_or_local_staging'))
        for name in sorted(filenames):
            source=Path(current)/name;relative=source.relative_to(ROOT);reason=excluded_reason(relative,keep)
            if not reason and source.stat().st_size>MAX_FILE_BYTES and relative.as_posix() not in keep and source.suffix.lower() not in ('.pdf','.pptx'):
                reason='individual_file_above_2_mib'
            if reason:
                if reason!='generated_publication_metadata':excluded.append(dict(path=relative.as_posix(),bytes=source.stat().st_size,reason=reason))
                continue
            if source.stat().st_size>=100*1024**2:raise RuntimeError(f'Unexpected file above regular Git size limit: {relative}')
            hits=scan_credentials(source)
            if hits:findings.append(dict(path=relative.as_posix(),patterns=hits))
            included.append(relative)
    if findings:
        # Deliberately report paths/pattern types only, never secret values.
        raise RuntimeError('Credential-pattern review required: '+json.dumps(findings))
    print('Scan passed; copying',len(included),'project files',flush=True)
    manifest={}
    for number,relative in enumerate(sorted(included),1):
        manifest[relative.as_posix()]=copy_exact(ROOT/relative,destination/relative)
        if number%1000==0:print('Copied and hashed',number,'files',flush=True)
    counts=Counter(r['reason'] for r in excluded)
    exclusion_report=dict(policy='Compact publication: no datasets or bulk arrays, redundant checkpoints/bundles/source copies; 2 MiB per-file cap except selected V2 checkpoints and existing PDF/PPTX/demo dependencies',
        selected_checkpoint_files=sorted(p for p in keep if p.endswith('/best.pt')),
        directory_exclusions=directories,file_exclusions=excluded,excluded_file_count=len(excluded),
        excluded_file_bytes=sum(r['bytes'] for r in excluded),reason_counts=dict(counts),source_files_deleted=False)
    security_report=dict(passed=True,scan='Known credential-prefix/private-key/credential-URL patterns in exported text files; no secret values logged',
        candidate_files=len(included),credential_pattern_findings=0,raw_datasets_included=False,maximum_file_bytes=max(v['bytes'] for v in manifest.values()))
    for name,value in [('publication_exclusions.json',exclusion_report),('publication_security.json',security_report)]:
        relative=Path('docs')/name;write_json(ROOT/relative,value)
        # Generated publication metadata is not an immutable research artifact.
        write_json(destination/relative,value)
        manifest[relative.as_posix()]=dict(bytes=(destination/relative).stat().st_size,sha256=digest(destination/relative))
    result=dict(schema_version=1,repository='https://github.com/Gopal1909/Visioners_2.0',snapshot_date='2026-10-26',
        files=dict(sorted(manifest.items())),file_count=len(manifest),total_bytes=sum(v['bytes'] for v in manifest.values()),
        self_hash_excluded=True,exclusions='docs/publication_exclusions.json',security='docs/publication_security.json',
        included_file_bytes_preserved=True,full_historical_archive_retained_in_original_workspace=True,
        compact_profile=True,large_training_caches_resumable=False)
    write_json(ROOT/'docs/publication_manifest.json',result);write_json(destination/'docs/publication_manifest.json',result)
    print(json.dumps({k:result[k] for k in ('file_count','total_bytes','repository')},indent=2))
    print('Excluded regenerable bytes:',exclusion_report['excluded_file_bytes'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();export(args.destination)
