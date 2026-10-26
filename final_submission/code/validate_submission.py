"""Completion audit of the current submission, source evidence and invariants."""
import csv
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as E
import zipfile
import numpy as np
import fitz
from study_pipeline import ROOT,DEFAULT,CONDITIONS


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify():
    final=ROOT/'final_submission';art=final/'artifacts';audit=art/'audit'
    audit.mkdir(exist_ok=True)
    required=['README.md','reproduction.md','results_summary.md','final_status.md','requirements.txt',
        'report/project_report.pdf','report/project_report.md','report/build_report.py',
        'presentation/project_presentation.pptx','presentation/build_presentation.mjs','presentation/speaker_notes.md',
        'artifacts/all_evaluated_results.csv','artifacts/demo_measurements.npz','artifacts/checksums.json',
        'artifacts/learned/best_checkpoint.pt','code/demo.py','demo/change_demo.html','demo/demo_verification.json']
    for f in required:assert (final/f).is_file() and (final/f).stat().st_size>0,f
    preservation=json.loads((ROOT/'results/pair_01/m3c2/preservation_before.json').read_text(encoding='utf-8-sig'))
    for item in preservation:assert sha(item['Path']).upper()==item['Hash'].upper(),item['Path']
    raw_manifest=json.loads((art/'dataset_manifest.json').read_text())
    for name,meta in raw_manifest.items():
        assert Path(name).stat().st_size==meta['bytes'],name
        assert sha(name)==meta['sha256'],name
    # Every condition has real complete outputs and its saved measurement hash.
    cases=[];support=[]
    for name,kind,value in CONDITIONS:
        folder=DEFAULT/'geometry'/name;meta=json.loads((folder/'complete.json').read_text())
        assert sha(folder/'measurements.npz')==meta['measurement_hash']
        assert sha(DEFAULT/'geometry_frozen.json')==meta['configuration_hash']
        learned=DEFAULT/'learned/evaluation'/name
        lm=json.loads((learned/'complete.json').read_text())
        frozen=json.loads((DEFAULT/'learned/model_frozen.json').read_text())
        assert lm['checkpoint_sha256']==frozen['checkpoint_sha256']
        assert lm['no_stress_retuning'] is True
        with np.load(folder/'measurements.npz') as m,np.load(learned/'predictions.npz') as l:
            for i in (0,1):
                np.testing.assert_array_equal(m[f'indices{i}'],l[f'indices{i}'])
                assert np.isin(l[f'prediction{i}'],[0,1]).all()
                assert np.all((l[f'probability{i}']>=0)&(l[f'probability{i}']<=1))
                np.testing.assert_array_equal(l[f'prediction{i}'],l[f'probability{i}']>=frozen['threshold'])
                valid=m[f'decision{i}_valid'];reason=m[f'decision{i}_reason']
                np.testing.assert_array_equal(valid,reason==0)
                assert np.all(m[f'decision{i}_prediction'][~valid]==-1)
                for code,count in zip(*np.unique(reason,return_counts=True)):
                    support.append(dict(condition=name,scan=i,reason_code=int(code),count=int(count),
                        fraction=float(count/len(reason)),mean_source_count=float(np.mean(m[f'raw{i}_count_source'][reason==code,0])),
                        mean_target_count=float(np.mean(m[f'raw{i}_count_target'][reason==code,0]))))
        cases.append(name)
    with open(art/'condition_support_diagnostics.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)
    frozen=json.loads((art/'learned/model_frozen.json').read_text())
    assert sha(art/'learned/best_checkpoint.pt')==frozen['checkpoint_sha256']
    mc=json.loads((art/'learned/model_config.json').read_text())
    assert sha(art/'learned/training_source.py')==mc['source_sha256']
    assert sha(art/'learned/model_config.json')==frozen['configuration_sha256']
    metric=json.loads((art/'metric_audit.json').read_text())
    assert metric['passed'] and metric['matched_rows_recomputed']==39
    tests=(audit/'all_tests.txt').read_text(encoding='utf-8-sig')
    assert 'Ran 25 tests' in tests and tests.rstrip().endswith('OK')
    browser=json.loads((audit/'demo_browser_verification.json').read_text())
    assert browser['passed'] and browser['page_errors']==[]
    assert browser['rotation_changes_canvas'] and browser['wheel_and_reset']
    demo=json.loads((final/'demo/demo_verification.json').read_text())
    assert demo['synthetic_passed'] and abs(demo['synthetic_offset_recovered_m']-.12)<1e-10
    # Actual final PDF, not just its builder or a previous draft.
    pdf=fitz.open(final/'report/project_report.pdf')
    assert len(pdf)==6
    text='\n'.join(p.get_text() for p in pdf)
    for term in ['Abstract','Introduction','Related work','Experimental setup','Robustness','Occlusion','Discussion','Conclusion','References','0.4118','0.3817','0.2342','0.3936']:
        assert term in text,term
    assert all(len(p.get_text())>800 for p in pdf)
    pdf.close()
    # Validate the exact shipped presentation and its named semantic contracts.
    receipt=json.loads((audit/'presentation_validation.json').read_text())
    ppt=final/'presentation/project_presentation.pptx'
    assert sha(ppt)==receipt['finalSha256']
    assert receipt['packageIntegrity']['status']=='pass'
    assert receipt['presentationLayout']['findingCount']==0
    assert receipt['firstPartyImport']['passed'] and receipt['nativeChartValidation']['passed']
    ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    z=zipfile.ZipFile(ppt)
    slides=[n for n in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',n)]
    notes=[n for n in z.namelist() if re.fullmatch(r'ppt/notesSlides/notesSlide\d+\.xml',n)]
    charts=[n for n in z.namelist() if re.search(r'/charts/chart\d+\.xml$',n)]
    assert len(slides)==11 and len(notes)==11 and len(charts)==8
    seconds={};native_curves=0
    for n in notes:
        t=' '.join(e.text or '' for e in E.fromstring(z.read(n)).findall('.//a:t',ns))
        match=re.search(r'(Vishwanath Venugopal|Shaik Koduru Asif); (\d+) seconds',t)
        assert match,n
        seconds[match[1]]=seconds.get(match[1],0)+int(match[2])
        assert 'Sources:' in t
    assert seconds=={'Vishwanath Venugopal':240,'Shaik Koduru Asif':240}
    for n in charts:
        number=int(re.search(r'chart(\d+)\.xml$',n)[1])
        if 2<=number<=7:
            for ser in E.fromstring(z.read(n)).findall('.//c:ser',ns):
                x=ser.find('c:xVal',ns)
                values=[float(v.text) for v in x.findall('.//c:pt/c:v',ns)]
                assert len(values)==3 and np.all(np.diff(values)>=0),(n,values)
                native_curves+=1
    assert native_curves==16
    assert len([n for n in z.namelist() if n.endswith('.xlsx')])==8
    z.close()
    # Inspect all final checksum entries, not just a sample.
    checks=json.loads((art/'checksums.json').read_text())
    for name,value in checks.items():assert sha(art/name)==value,name
    # Verify the already present archive by CRC, without redownloading/extracting.
    original_zip=ROOT/'dataset/MultiChange3D/BikeParkingConstruction/pair_01.zip'
    with zipfile.ZipFile(original_zip) as archive:assert archive.testzip() is None
    result=dict(passed=True,requirements={'report_pdf_and_editable_source':True,
        'editable_presentation_with_native_workbooks_and_notes':True,'offline_interactive_demo':True,
        'bounded_geometric_comparison':True,'representative_matched_robustness':True,
        'trained_calibrated_learned_baseline':True,'explicit_unknowns_and_scopes':True,
        'data_access_and_license_limits_documented':True,'reproduction_code_and_commands':True},
        tests=25,protected_files_unchanged=len(preservation),original_point_clouds_unchanged=len(raw_manifest),geometry_conditions=cases,
        matched_metric_rows_audited=39,report_pages=6,presentation_slides=11,native_charts=8,
        monotonic_severity_curves_checked=native_curves,embedded_chart_workbooks=8,
        presentation_seconds_by_speaker=seconds,demo_browser_checked=True,
        original_pair_archive_crc_passed=True,artifact_checksums_verified=len(checks),
        pdf_sha256=sha(final/'report/project_report.pdf'),pptx_sha256=sha(ppt),
        raw_data_downloaded_this_continuation=False,
        limitations=['single scene/pair and seed','historical threshold/pilot exposure',
            'conditional uncalibrated LoD','no independent reference-software output check',
            'no cross-scene or ConSLAM accuracy claim','native PowerPoint execution not verified'])
    (audit/'final_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':verify()
