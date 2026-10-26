"""Describe false alarms, missed changes and abstentions without retuning models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from multipair_data import DEFAULT


def diagnose(out=DEFAULT):
    out=Path(out);folder=out/'analysis';folder.mkdir(exist_ok=True)
    rows=[];unknown=[]
    roots=[out,*sorted((out/'spatial_cv').glob('fold_*'))] if (out/'spatial_cv').exists() else [out]
    for root in roots:
        for path in (root/'evaluation').glob('*/*/clean_seed42/predictions.npz'):
            with np.load(path) as z:
                for scan in (0,1):
                    y=z[f'labels{scan}'];pred=z[f'm3c2_calibrated_prediction{scan}']
                    prefix=f'm3c2_calibrated_scan{scan}_'
                    reason=z[prefix+'decision_reason']
                    distance=np.abs(z[prefix+'signed_distance_m'][:,0]);lod=z[prefix+'decision_lod95_m']
                    source=z[prefix+'count_source'][:,0];target=z[prefix+'count_target'][:,0]
                    groups={'TP':(y==1)&(pred==1),'FP':(y==0)&(pred==1),'FN_valid':(y==1)&(pred==0),
                        'TN':(y==0)&(pred==0),'unknown_changed':(y==1)&(pred<0),'unknown_unchanged':(y==0)&(pred<0)}
                    for group,mask in groups.items():
                        def median(values):
                            a=values[mask];a=a[np.isfinite(a)]
                            return float(np.median(a)) if len(a) else None
                        rows.append(dict(run=str(root.relative_to(out)) if root!=out else 'main',
                            experiment=path.parents[2].name,pair=path.parents[1].name,scan=scan,group=group,count=int(mask.sum()),
                            median_abs_displacement_m=median(distance),median_lod95_m=median(lod),
                            median_source_support=median(source),median_target_support=median(target)))
                    for label in (0,1):
                        for code,n in zip(*np.unique(reason[y==label],return_counts=True)):
                            unknown.append(dict(run=str(root.relative_to(out)) if root!=out else 'main',
                                experiment=path.parents[2].name,pair=path.parents[1].name,scan=scan,label=label,reason_code=int(code),count=int(n)))
    pd.DataFrame(rows).to_csv(folder/'m3c2_error_diagnostics.csv',index=False)
    pd.DataFrame(unknown).to_csv(folder/'m3c2_reason_by_true_class.csv',index=False)
    plot_error_maps(out)


def plot_error_maps(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    out=Path(out);folder=out/'analysis/error_maps';folder.mkdir(parents=True,exist_ok=True)
    selected=json.loads((out/'learning/primary/selection_frozen.json').read_text())['selected_variant']
    methods=[('GT','Ground truth'),('c2c_calibrated','C2C calibrated'),('m3c2_calibrated','M3C2-style'),
        (selected+'_seed42','PointNet + NN'),('geometry_boosted','Geometry classifier')]
    palette={'TP':'#16a34a','FP':'#dc2626','FN':'#f59e0b','unknown':'#8b5cf6','TN':'#cbd5e1'}
    for path in sorted((out/'evaluation/primary').glob('*/clean_seed42/predictions.npz')):
        with np.load(path) as z:
            xyz=[z[f'xyz{i}'] for i in (0,1)];labels=[z[f'labels{i}'] for i in (0,1)]
            allxyz=np.concatenate(xyz);lo=allxyz.min(0);hi=allxyz.max(0)
            fig=plt.figure(figsize=(18,8))
            for scan in (0,1):
                for column,(method,title) in enumerate(methods):
                    ax=fig.add_subplot(2,len(methods),scan*len(methods)+column+1,projection='3d')
                    y=labels[scan]
                    if method=='GT':colors=np.where(y==1,'#dc2626','#94a3b8')
                    else:
                        p=z[f'{method}_prediction{scan}'];colors=np.full(len(y),palette['TN'])
                        colors[(y==1)&(p==1)]=palette['TP'];colors[(y==0)&(p==1)]=palette['FP']
                        colors[(y==1)&(p==0)]=palette['FN'];colors[p<0]=palette['unknown']
                    ax.scatter(*xyz[scan].T,c=colors,s=2,alpha=.8,depthshade=False)
                    ax.set_xlim(lo[0]-.1,hi[0]+.1);ax.set_ylim(lo[1]-.1,hi[1]+.1);ax.set_zlim(lo[2]-.1,hi[2]+.1)
                    ax.set_box_aspect(np.maximum(hi-lo,.1));ax.view_init(27,-57)
                    ax.set_xticks(np.round([lo[0],hi[0]],1))
                    ax.set_yticks(np.round([lo[1],(lo[1]+hi[1])/2,hi[1]],1))
                    ax.set_zticks(np.round([lo[2],hi[2]],1))
                    ax.set_xlabel('X (m)',fontsize=7);ax.set_ylabel('Y (m)',fontsize=7);ax.set_zlabel('Z (m)',fontsize=7)
                    ax.tick_params(labelsize=6);ax.set_title(title+f' | scan {scan}',fontsize=10)
            handles=[Line2D([],[],marker='o',linestyle='',color=color,label=name) for name,color in palette.items()]
            fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=10)
            fig.suptitle(path.parents[1].name+' | fold 0 clean held-out original cores (not a full-cloud map)\nGround truth: red = changed; gray = unchanged. Other panels show prediction errors.',fontsize=12)
            fig.subplots_adjust(left=.03,right=.98,top=.88,bottom=.12,wspace=.12,hspace=.08)
            fig.savefig(folder/(path.parents[1].name+'.png'),dpi=160);plt.close(fig)


if __name__=='__main__':diagnose()
