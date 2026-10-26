"""Assemble bounded evidence and reproducible code, excluding original datasets."""
import hashlib
import json
from pathlib import Path
import shutil
import importlib.metadata
from study_pipeline import ROOT,DEFAULT


def run():
    final=ROOT/'final_submission'
    for name in ('artifacts','code','report','presentation','demo'):(final/name).mkdir(parents=True,exist_ok=True)
    artifacts=final/'artifacts'
    def copy(src,relative):
        dst=artifacts/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for src in (DEFAULT/'analysis').iterdir():
        if src.is_file():copy(src,src.name if src.suffix!='.png' else 'figures/'+src.name)
    for rel in ('split_config.json','dataset_manifest.json','geometry_frozen.json','geometry_smoke.json','perturbation_geometry.json','geometry_metrics.csv'):
        copy(DEFAULT/rel,rel)
    for rel in ('model_config.json','model_frozen.json','best_checkpoint.pt','training_history.csv','training_runtime.json','calibration_thresholds.csv','evaluation_metrics.csv'):
        copy(DEFAULT/'learned'/rel,'learned/'+rel)
    # Archive the actual training-source version before the later evaluation-only
    # metadata addition. Verify its hash against the recorded training config.
    source=(ROOT/'code/siamese_pointnet.py').read_text()
    start=source.index("    write_json(out/'perturbation_geometry.json'")
    end=source.index('    with MemoryTimer() as timer: tree0=',start)
    original=source[:start]+source[end:]
    expected=json.loads((DEFAULT/'learned/model_config.json').read_text())['source_sha256']
    assert hashlib.sha256(original.encode()).hexdigest()==expected
    # Keep the exact UTF-8/LF bytes used by the recorded source digest on Windows.
    (artifacts/'learned/training_source.py').write_bytes(original.encode('utf-8'))
    for rel in ('geometry_metrics.csv','geometry_frozen.json'):
        copy(ROOT/'results/occlusion_supplement_2026-10-25_v1'/rel,'occlusion_supplement/'+rel)
    for name in ('inspection.json','conslam_qualitative.png','conslam_trajectory.png'):
        copy(ROOT/'results/conslam_qualitative_2026-10-25_v2'/name,
             'figures/'+name if name.endswith('.png') else 'conslam/'+name)
    for rel in ('primary_comparison.csv','pilot_summary.md','reference_conventions.json','final_verification.json'):
        copy(ROOT/'results/pair_01/m3c2/pilot_2026-10-25_v2'/rel,'historical_pilot/'+rel)
    copy(ROOT/'results/master_audit_2026-10-25/audit.json','audit/saved_results_and_geometry.json')
    copy(ROOT/'results/pair_01/m3c2/preservation_verified.json','audit/historical_preservation.json')
    for src in (ROOT/'results/pair_01/robustness_reviewed').glob('*metrics.csv'):
        copy(src,'historical_full_pair/'+src.name)
    copy(ROOT/'results/pair_01/c2c_baseline/threshold_metrics.csv','historical_full_pair/c2c_threshold_metrics.csv')
    copy(ROOT/'results/data_sources_and_access.md','data_sources_and_access.md')
    for src in (ROOT/'code').glob('*.py'):shutil.copy2(src,final/'code'/src.name)
    versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','plyfile','pandas','psutil','matplotlib','torch','open3d')}
    (final/'environment_versions.json').write_text(json.dumps(versions,indent=2))
    (final/'requirements.txt').write_text('\n'.join(f'{p}=={v}' for p,v in versions.items())+'\n')
    checks={str(p.relative_to(artifacts)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in artifacts.rglob('*') if p.is_file() and p.name!='checksums.json'}
    (artifacts/'checksums.json').write_text(json.dumps(checks,indent=2))
    print('Packaged evidence files:',len(checks),'bytes',sum(p.stat().st_size for p in final.rglob('*') if p.is_file()))


if __name__=='__main__':run()
