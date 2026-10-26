"""Readable maps from frozen region exports; no regrouping or score changes."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from learning_v3_control import ROOT,digest,write_json


def render():
    root=ROOT/'results/construction_change_regions_2026-10-26_v1';dst=root/'figures_readable';dst.mkdir(exist_ok=True)
    rows=list(csv.DictReader((root/'candidate_regions.csv').open()));manifest={}
    for pair in ('01','02','03','12','13','23'):
        fig,axes=plt.subplots(1,2,figsize=(13,6),sharex=True,sharey=True)
        for ax,kind in zip(axes,('removed','added')):
            with np.load(root/f'pair_{pair}_{kind}_cores.npz') as z:
                xyz=z['xyz'];membership=z['component_index'];parts=[r for r in rows if r['pair']==pair and r['kind']=='candidate_'+kind]
                ax.scatter(xyz[:,0],xyz[:,1],s=2,c='#cbd5e1',alpha=.6);positions=[]
                lo=xyz[:,:2].min(0);hi=xyz[:,:2].max(0)
                for r in parts:
                    component=int(r['region_id'].rsplit('_',1)[1])-1;points=xyz[membership==component]
                    c=points.mean(0);a=points.min(0);b=points.max(0);color=plt.get_cmap('turbo')(component/max(1,len(parts)-1))
                    ax.scatter(points[:,0],points[:,1],s=6,color=color)
                    if component<8:
                        ax.add_patch(Rectangle(a[:2],b[0]-a[0],b[1]-a[1],fill=False,color=color,lw=1))
                        offsets=[(.25,.25),(.25,.8),(.25,-.8),(-1.,.25),(-1.,-.8),(1.,.8),(-1.,.8),(1.5,-.8)]
                        candidates=[c[:2]+offset for offset in offsets]
                        candidates=[p for p in candidates if (p>=lo+.1).all() and (p<=hi-.1).all()]
                        candidates=[p for p in candidates if all(abs(p[0]-q[0])>.8 or abs(p[1]-q[1])>.5 for q in positions)] or [c[:2]+(.25,.25)]
                        position=candidates[0];positions.append(position)
                        ax.annotate(f'{component+1:03d}',xy=c[:2],xytext=position,fontsize=8,color='#111827',
                            bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=.5),arrowprops=dict(arrowstyle='-',color='#475569',lw=.6))
                residual=membership==-1;ax.scatter(xyz[residual,0],xyz[residual,1],s=7,c='#dc2626',marker='x',label='Ungrouped positives')
                ax.set_title(f'Candidate {kind} geometry | {len(parts)} regions');ax.set_aspect('equal');ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.legend(fontsize=8,loc='lower left')
        fig.suptitle(f'Construction pair_{pair}: epoch {pair[0]} to {pair[1]} | sampled candidate change regions\n7500 cores/scan across five folds; gaps unscored. Numbers are region-ID suffixes; XYZ bounds are in the CSV.',fontsize=11)
        fig.subplots_adjust(left=.055,right=.98,bottom=.12,top=.82,wspace=.18)
        path=dst/f'pair_{pair}_regions.png';fig.savefig(path,dpi=170);plt.close(fig);manifest[str(path.relative_to(root))]=digest(path)
    write_json(dst/'manifest.json',dict(renderer_sha256=digest(__file__),figures=manifest,source_predictions_and_membership_unchanged=True))


if __name__=='__main__':render()
