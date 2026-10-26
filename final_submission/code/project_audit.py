"""Independent saved-result and streamed real-cylinder audit (no full experiments)."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from c2c_baseline import load_point_cloud_ply
from m3c2_baseline import digest, write_json


def run(out):
    root=Path(__file__).resolve().parents[1]
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    result={'stage3':{},'m3c2':{},'sources':{}}
    folder=root/'results/pair_01/robustness_reviewed'
    worst=0.; checked=0; rows=0
    for f in sorted(folder.glob('*metrics.csv')):
        df=pd.read_csv(f)
        for _,r in df.iterrows():
            if 'scan0_TP' not in r: continue
            confusion=[]
            for i in (0,1):
                tp,fp,fn,tn=[r[f'scan{i}_{k}'] for k in ('TP','FP','FN','TN')]
                p=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
                expected={'precision':p,'recall':rec,'f1':2*tp/max(1,2*tp+fp+fn),
                          'iou_changed':tp/max(1,tp+fp+fn),
                          'mIoU':.5*(tp/max(1,tp+fp+fn)+tn/max(1,tn+fp+fn))}
                for k,v in expected.items():
                    if f'scan{i}_{k}' in r:
                        worst=max(worst,abs(v-r[f'scan{i}_{k}'])); checked+=1
                confusion.append((tp,fp,fn,tn))
            (a,b,c,d),(e,f0,g,h)=confusion
            mat=np.array([[d+h,b,f0],[c,a,0],[g,0,e]])
            tp=np.diag(mat); fp=mat.sum(0)-tp; fn=mat.sum(1)-tp
            f1=2*tp/np.maximum(1,2*tp+fp+fn); iou=tp/np.maximum(1,tp+fp+fn)
            for key,v in [('comb_macro_f1',f1.mean()),('comb_macro_mIoU',iou.mean()),
                          ('comb_overall_acc',tp.sum()/mat.sum())]:
                if key in r:
                    worst=max(worst,abs(v-r[key]));checked+=1
            rows+=1
        result['stage3'][f.name]={'rows':len(df),'sha256':digest(f)}
    assert worst<1e-7
    result['stage3'].update(confusion_rows_checked=rows,metric_values_checked=checked,
        max_abs_error=worst,report_exists=(folder/'robustness_report_final.txt').exists(),
        occlusion_limit='Aggregate CSV lacks confusion counts; not independently reconstructed')
    pilot=root/'results/pair_01/m3c2/pilot_2026-10-25_v2'
    m=pd.read_csv(pilot/'primary_comparison.csv')
    result['m3c2']['saved_primary']=m[['method','combined_macro_f1','combined_macro_mIoU','coverage','unknown_rate']].to_dict('records')
    np.testing.assert_allclose(m.combined_macro_f1,[.35288027,.62741661],atol=1e-6)
    np.testing.assert_allclose(m.coverage,.6335)
    cores=dict(np.load(pilot/'selected_cores.npz'))
    clouds=[load_point_cloud_ply(root/f'dataset/MultiChange3D/BikeParkingConstruction/pair_01/pointCloud{i}.ply') for i in (0,1)]
    checks=[]
    for i in (0,1):
        normal=dict(np.load(pilot/f'normals_scan{i}.npz'))
        raw=dict(np.load(pilot/f'raw_scan{i}.npz'))
        good=np.flatnonzero(normal['normal_reason']==0)
        for a in good[np.linspace(0,len(good)-1,3,dtype=int)]:
            c=cores[f'xyz{i}'][a]; n=normal['normals'][a]
            # Independent complete-cloud streaming, no spatial tree or enclosing sphere.
            sums=[]
            for epoch in (i,1-i):
                count=0; total=0.; total2=0.
                for start in range(0,len(clouds[epoch]['xyz']),100000):
                    rel=clouds[epoch]['xyz'][start:start+100000].astype('float64')-c
                    axial=np.sum(rel*n,axis=1)
                    radial2=np.sum(np.cross(rel,n)**2,axis=1)
                    t=axial[(radial2<=.10**2)&(np.abs(axial)<=1.)]
                    count+=len(t);total+=t.sum();total2+=np.square(t).sum()
                sums.append((count,total/count if count else np.nan,
                             np.sqrt(max(0,total2/count-(total/count)**2)) if count else np.nan))
            for tag,(count,mean,sd) in zip(('source','target'),sums):
                assert count==raw[f'count_{tag}'][a,1]
                if count>=10:
                    np.testing.assert_allclose(mean,raw[f'mean_{tag}_m'][a,1],atol=1e-10)
                    np.testing.assert_allclose(sd,raw[f'spread_{tag}_m'][a,1],atol=1e-9)
            checks.append(dict(scan=i,original_index=int(cores[f'indices{i}'][a]),
                               complete_cylinder_counts=[s[0] for s in sums]))
    result['m3c2']['independent_full_support_streaming_checks']=checks
    result['m3c2']['assessment']='No geometry or LoD bug found in tested paths. Cross-scale criterion differs from CloudCompare; retain M3C2-style name.'
    result['sources']={str(p.relative_to(root)):digest(p) for p in (root/'code').glob('*.py')}
    write_json(out/'audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/master_audit_2026-10-25')
    run(p.parse_args().out)
