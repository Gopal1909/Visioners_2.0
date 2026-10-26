"""Finish human-readable evidence, refresh checksums and package the verified work."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from package_submission import run as package
from study_pipeline import ROOT,DEFAULT


def rows(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def run():
    package()
    final=ROOT/'final_submission';art=final/'artifacts';audit=art/'audit';audit.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'README.md',final/'reproduction.md')
    shutil.copy2(ROOT/'results/final_status.md',final/'final_status.md')
    shutil.copy2(ROOT/'code/build_report.py',final/'report/build_report.py')
    shutil.copy2(ROOT/'results/submission_build/slides/validation_final.json',audit/'presentation_validation.json')
    shutil.copy2(ROOT/'results/submission_build/demo_browser_verification.json',audit/'demo_browser_verification.json')
    shutil.copy2(ROOT/'results/master_logs/all_tests.txt',audit/'all_tests.txt')
    metric=rows(art/'matched_condition_metrics.csv')
    methods=['C2C common','M3C2-style','Siamese PointNet common']
    text=['# Measured results by evaluation scope\n',
        'The table below contains condition-specific common-valid scores on the new buffered same-scene spatial-test cores. Parameters stay fixed. C2C and learned exclusions follow the M3C2 validity mask for these rows. Full precision, directional precision/recall/F1/IoU, coverage, runtime and memory are in `artifacts/all_evaluated_results.csv`.\n',
        '| Condition | C2C macro F1 | M3C2-style macro F1 | Learned + NN macro F1 | Common coverage |\n|---|---:|---:|---:|---:|']
    for condition in dict.fromkeys(r['condition'] for r in metric):
        entries={r['method']:r for r in metric if r['condition']==condition and r['protocol'] in ('standard','A')}
        values=[float(entries[k]['combined_macro_f1']) for k in methods]
        text.append(f'| {condition} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {float(entries[methods[0]]["coverage"]):.2%} |')
    text += ['\nThe main spatial stripe misses both deletion cylinders. Its identical occlusion scores are **not** occlusion robustness evidence. The separate exploratory all-scene geometric supplement supplies strict A and visibility-masked B results; do not combine that scope into the table\'s ranking.\n',
        'At 7.6 m deletion radius, matched C2C/M3C2 macro F1 is 0.5925/0.3066 under A and 0.5950/0.3074 under B. Common coverage is 51.95% / 67.04%. The learned model is not scored on supplement training-region points as a held-out result.\n',
        'Historical full-pair C2C macro F1 0.7359/mIoU 0.6046 and the earlier all-scene pilot C2C/M3C2 macro F1 0.6274/0.3529 are separate exploratory references. The historical threshold was selected on that pair. No cross-pair or cross-scene result exists.\n',
        'For the learned model alone, calibration all-core macro F1 0.7649 versus spatial-test all-core 0.3936 exposes a large spatial gap. Its common-support score 0.3817 belongs to the three-method table above. All-core C2C macro F1 is 0.4457.\n',
        'Use `artifacts/matched_clean_deltas.csv` for longitudinal changes on identical clean/current valid retained original indices. These isolate support changes more carefully than subtracting differently masked table rows.\n',
        'Unknowns remain explicit. M3C2 clean unknown rate is 33.125%; C2C and learned classifiers have zero detector abstentions. The 66.875% common evaluation coverage is an evaluation restriction for those two classifiers.\n']
    (final/'results_summary.md').write_text('\n'.join(text),encoding='utf-8')
    # The final audit writes a new support diagnostic before checksums; precreate
    # that table here so the demo checks a complete, stable artifact inventory.
    diagnostics=[]
    import numpy as np
    from study_pipeline import CONDITIONS
    for name,_,_ in CONDITIONS:
        with np.load(DEFAULT/f'geometry/{name}/measurements.npz') as m:
            for i in (0,1):
                reason=m[f'decision{i}_reason']
                for code,count in zip(*np.unique(reason,return_counts=True)):
                    diagnostics.append(dict(condition=name,scan=i,reason_code=int(code),count=int(count),fraction=float(count/len(reason)),
                        mean_source_count=float(np.mean(m[f'raw{i}_count_source'][reason==code,0])),
                        mean_target_count=float(np.mean(m[f'raw{i}_count_target'][reason==code,0]))))
    with open(art/'condition_support_diagnostics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostics[0]));w.writeheader();w.writerows(diagnostics)
    checksum={str(p.relative_to(art)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in art.rglob('*') if p.is_file() and p.name not in ('checksums.json','final_verification.json')}
    (art/'checksums.json').write_text(json.dumps(checksum,indent=2))
    # Use the packaged code/measurements, not a private in-memory demo state.
    import subprocess,sys
    subprocess.run([sys.executable,str(final/'code/demo.py')],check=True,cwd=ROOT)
    from validate_submission import verify
    verify()
    manifest={str(p.relative_to(final)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in final.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='submission_manifest.json'}
    (final/'submission_manifest.json').write_text(json.dumps(manifest,indent=2))
    archive=ROOT/'final_submission_bundle.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(final.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,Path('final_submission')/p.relative_to(final))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print('Verified bundle:',archive,'bytes',archive.stat().st_size,'files',len(manifest)+1)


if __name__=='__main__':run()
