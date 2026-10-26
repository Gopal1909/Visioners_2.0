"""Publication-sized plots of measured evidence, without modifying past figures."""
from pathlib import Path
import json
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study_pipeline import ROOT,DEFAULT
from analyze_study import COLORS,NAMES,ordered_curve


def run():
    dst=ROOT/'final_submission/artifacts/figures';dst.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    common=pd.read_csv(DEFAULT/'analysis/matched_condition_metrics.csv')
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4))
    for ax,(family,label,scale) in zip(axes.flat,[('translation','Translation (cm)',100),('rotation','Rotation (degrees)',1),('sparsity','Scan 1 retained (%)',100),('noise','Noise SD per axis (cm)',100)]):
        for method,color in COLORS.items():
            base=common[(common.condition=='clean')&(common.method==method)].iloc[0]
            group=common[(common.family==family)&(common.method==method)].sort_values('severity')
            xx,yy=ordered_curve([100 if family=='sparsity' else 0]+(group.severity*scale).tolist(),
                               [base.combined_macro_f1]+group.combined_macro_f1.tolist())
            ax.plot(xx,yy,'o-',ms=3,lw=1.5,
                    color=color,label=NAMES[method])
        ax.set(xlabel=label,ylabel='Macro F1',ylim=(0,.65));ax.grid(alpha=.2)
        if family=='sparsity':ax.invert_xaxis()
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=8.5,frameon=False)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(dst/'paper_robustness.png',dpi=220);plt.close(fig)
    supp=pd.read_csv(ROOT/'results/occlusion_supplement_2026-10-25_v1/geometry_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.65))
    for protocol,ax in zip(('A','B'),axes):
        for method,color in list(COLORS.items())[:2]:
            base=supp[(supp.condition=='clean')&(supp.method==method)].iloc[0]
            group=supp[(supp.protocol==protocol)&(supp.method==method)].sort_values('severity')
            ax.plot([0]+group.severity.tolist(),[base.combined_macro_f1]+group.combined_macro_f1.tolist(),'o-',ms=3,color=color,label=NAMES[method])
        ax.set(title='A: strict' if protocol=='A' else 'B: known blind region excluded',xlabel='Deletion radius (m)',ylabel='Macro F1',ylim=(0,.8));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,frameon=False);fig.tight_layout();fig.savefig(dst/'paper_occlusion.png',dpi=220);plt.close(fig)
    m=dict(np.load(DEFAULT/'geometry/clean/measurements.npz'));l=dict(np.load(DEFAULT/'learned/evaluation/clean/predictions.npz'))
    fig,axes=plt.subplots(1,4,figsize=(7.2,2.9),sharex=True,sharey=True)
    p=m['xyz1'];valid=m['decision1_valid']
    for ax,title,v in zip(axes,['Truth','C2C','M3C2-style','Learned'],[m['gt1'],m['c2c_prediction1'],m['decision1_prediction'],l['prediction1']]):
        v=v if title=='Truth' else np.where(valid,v,-1)
        ax.scatter(p[:,0],p[:,1],s=1,c=np.where(v<0,'#9ca3af',np.where(v==1,'#dc2626','#2563eb')))
        ax.set(title=title,xlabel='X (m)');ax.set_aspect('equal')
    axes[0].set_ylabel('Y (m)');fig.tight_layout();fig.savefig(dst/'paper_change_maps.png',dpi=220);plt.close(fig)
    # Scene overview uses real all-scene original cores with verified change labels.
    with np.load(ROOT/'results/pair_01/m3c2/pilot_2026-10-25_v2/selected_cores.npz') as z:
        p=z['xyz0'];y=z['cd_type0']
    fig=plt.figure(figsize=(7,4.5));ax=fig.add_subplot(111,projection='3d')
    ax.scatter(p[:,0],p[:,1],p[:,2],s=4,c=np.where(y==1,'#dc2626','#2563eb'),alpha=.85)
    ax.set_box_aspect(np.maximum(np.ptp(p,axis=0),.1));ax.set_axis_off()
    fig.tight_layout();fig.savefig(dst/'site_overview.png',dpi=200,transparent=True);plt.close(fig)
    info=json.loads((ROOT/'results/conslam_qualitative_2026-10-25_v2/inspection.json').read_text())
    fig,ax=plt.subplots(figsize=(7.2,2.3))
    for record,color in zip(info['selected_frames'],['#2563eb','#ea580c']):
        xyz=np.asarray(o3d.io.read_point_cloud(record['pcd']).points)
        t=np.array(record['pose_matrix']);xyz=xyz@t[:3,:3].T+t[:3,3]
        idx=np.linspace(0,len(xyz)-1,min(6000,len(xyz)),dtype=int)
        ax.scatter(xyz[idx,0],xyz[idx,1],c=color,s=.4,alpha=.7)
    ax.set(xlabel='X (m)',ylabel='Y (m)');ax.set_aspect('equal',adjustable='datalim')
    fig.tight_layout();fig.savefig(dst/'paper_conslam.png',dpi=220);plt.close(fig)
    print('Publication figures saved',dst)


if __name__=='__main__':run()
