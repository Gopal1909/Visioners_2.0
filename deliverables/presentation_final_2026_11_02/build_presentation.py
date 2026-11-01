"""CV5570 evidence-first presentation. Run with dl_env Python; PowerPoint is required for export.
Only writes beside this script. Original data and historical outputs are read-only.
"""
from pathlib import Path
import csv, json, hashlib, shutil, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
import fitz

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
ASSETS=OUT/'assets'; RENDERS=OUT/'slide_renders'
for d in (ASSETS,RENDERS): d.mkdir(exist_ok=True)
STUDY=ROOT/'results'/'learning_v2_coverage_2026-10-25_v1'
REG=ROOT/'results'/'construction_change_regions_2026-10-26_v1'
BG='0B1322'; PANEL='132236'; WHITE='F3F6FA'; MUTED='AAB8C9'; TEAL='49D4CB'; ORANGE='F5A360'; PURPLE='B49AF5'; GREY='667889'
provenance=[]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def track(p,asset=None,detail=''):
 p=Path(p);provenance.append(dict(source=str(p),source_sha256=sha(p),asset=str(asset) if asset else None,detail=detail))
def rows(p):
 track(p);return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
clean=rows(STUDY/'analysis'/'clean_comparison.csv');metrics={r['method']:r for r in clean}
stress=rows(STUDY/'analysis'/'stress_condition_summary.csv')
pooled=rows(STUDY/'analysis'/'class_support_and_pooled_confusions.csv')
regions=rows(REG/'candidate_regions.csv')
assert len(regions)==191
assert sum(int(r['evaluation_changed_cores'])==0 for r in regions)==83
assert abs(float(metrics['v2_validation_selected']['binary_macro_f1'])-.6536591206208738)<1e-12
protocol=json.loads((STUDY/'fold_0'/'protocol.json').read_text());track(STUDY/'fold_0'/'protocol.json')
assert protocol['core_boundary_guard_m']==2.5
track(STUDY/'design.json');track(ROOT/'code'/'learning_v2_model.py')
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#'+WHITE,'axes.labelcolor':'#'+MUTED,'xtick.color':'#'+MUTED,'ytick.color':'#'+MUTED,'axes.edgecolor':'#'+GREY,'font.size':14,'savefig.facecolor':'#'+BG,'figure.facecolor':'#'+BG,'axes.facecolor':'#'+BG})

def ply_sample(p,n=130000):
 with p.open('rb') as f:
  header=b''
  while not header.endswith(b'end_header\n'):header+=f.readline()
 count=int([x for x in header.decode().splitlines() if x.startswith('element ')][0].split()[-1])
 a=np.memmap(p,mode='r',dtype='<f4',offset=len(header),shape=(count,5))
 ids=np.linspace(0,count-1,min(n,count)).astype(int)
 pts=np.asarray(a[ids,:3]);lab=np.asarray(a[ids,4]);track(p,detail=f'Uniform index-stride display sample {len(ids):,}/{count:,}; native XYZ, no inference')
 return pts,lab
pair=ROOT/'dataset'/'MultiChange3D'/'BikeParkingConstruction'/'pair_03'/'pair_03'
early,elab=ply_sample(pair/'pointCloud0.ply');late,llab=ply_sample(pair/'pointCloud1.ply')
center=np.concatenate([early,late]).mean(0)
def project(x):
 x=x-center
 return np.column_stack((.84*x[:,0]-.54*x[:,1],.31*x[:,0]+.48*x[:,1]+1.7*x[:,2]))
lim=np.percentile(np.vstack((project(early),project(late))),[.1,99.9],axis=0)
def cloud_image(x,lab,name,color):
 fig,ax=plt.subplots(figsize=(9,5),dpi=190);v=project(x)
 ax.scatter(v[:,0],v[:,1],c=np.where(lab>0,'#'+color,'#8D9CAF'),s=.12,alpha=.72,rasterized=True,linewidths=0)
 ax.set(xlim=(lim[0,0],lim[1,0]),ylim=(lim[0,1],lim[1,1]));ax.set_aspect('equal');ax.axis('off');fig.subplots_adjust(0,0,1,1)
 p=ASSETS/name;fig.savefig(p);plt.close(fig);return p
cloud_image(early,elab,'epoch_0.png',ORANGE);cloud_image(late,llab,'epoch_3.png',TEAL)
# Scan-error illustrations use the genuine epoch-3 sample, not stock or fabricated geometry.
v=project(late[::3]);rng=np.random.default_rng(42)
fig,axs=plt.subplots(1,4,figsize=(14,4),dpi=170)
for ax,title,mode in zip(axs,['Misregistration','Noise','Sparsity','Missing view'],range(4)):
 ax.scatter(v[:,0],v[:,1],s=.12,c='#'+GREY,alpha=.45,linewidths=0)
 z=v.copy()
 if mode==0:z+=np.array([.8,.55])
 elif mode==1:z+=rng.normal(0,.16,z.shape)
 elif mode==2:z=z[rng.random(len(z))<.13]
 else:z=z[z[:,0]>np.quantile(z[:,0],.5)]
 ax.scatter(z[:,0],z[:,1],s=.18,c='#'+TEAL,alpha=.6,linewidths=0);ax.axis('off');ax.set_aspect('equal');ax.set_title(title,color='#'+WHITE,pad=15)
fig.tight_layout();fig.savefig(ASSETS/'scan_errors.png',bbox_inches='tight');plt.close(fig)
# Actual sampled held-out predictions, consistent semantic colors.
a=np.load(REG/'pair_03_added_cores.npz');track(REG/'pair_03_added_cores.npz',detail='All 7500 later-epoch out-of-fold sampled cores, pair 03; display-only error classifications')
xyz=a['xyz'];y=a['evaluation_labels'];pred=a['prediction'];s=a['model_score'];xy=project(xyz)
fig,axs=plt.subplots(1,3,figsize=(14,4.8),dpi=180)
for ax,title,mask,c in [(axs[0],'False alarms (FP)',(pred==1)&(y==0),ORANGE),(axs[1],'Missed change (FN)',(pred==0)&(y==1),TEAL),(axs[2],'Near decision boundary',np.abs(s-.5)<.1,PURPLE)]:
 ax.scatter(xy[:,0],xy[:,1],s=.7,c='#'+GREY,alpha=.3,linewidths=0)
 ax.scatter(xy[mask,0],xy[mask,1],s=7,c='#'+c,alpha=.9,linewidths=0)
 ax.set_title(f'{title}\n{int(mask.sum())} displayed cores',color='#'+WHITE,pad=12);ax.axis('off');ax.set_aspect('equal')
fig.tight_layout();fig.savefig(ASSETS/'uncertainty_atlas.png',bbox_inches='tight');plt.close(fig)
# Regenerate region map from candidate CSV, sized by sampled membership; no objects implied.
fig,ax=plt.subplots(figsize=(9,4.8),dpi=180)
rr=[r for r in regions if r['pair']=='03']
for kind,col in [('candidate_added',TEAL),('candidate_removed',ORANGE)]:
 r=[x for x in rr if x['kind']==kind]
 ax.scatter([float(x['centroid_x']) for x in r],[float(x['centroid_y']) for x in r],s=[18+4*np.sqrt(int(x['sampled_positive_cores'])) for x in r],c='#'+col,alpha=.8,label=kind.replace('candidate_','').title(),edgecolors='none')
ax.set(xlabel='Site X (m)',ylabel='Site Y (m)');ax.spines[['top','right']].set_visible(False);ax.legend(facecolor='#'+PANEL,labelcolor='white',frameon=False);fig.tight_layout();fig.savefig(ASSETS/'region_centroids.png');plt.close(fig)
cp=ROOT/'results'/'conslam_qualitative_2026-10-25_v2'/'conslam_qualitative.png'
shutil.copy2(cp,ASSETS/'conslam_qualitative.png');track(cp,ASSETS/'conslam_qualitative.png','Original local qualitative figure; unmodified')
track(ROOT/'results'/'conslam_qualitative_2026-10-25_v2'/'inspection.json')

prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
notes=[];titles=[];durations=[30,40,40,40,45,55,50,55,50,45,45,35,40,30]
assert sum(durations)==600 and sum(durations[:7])==300

def rgb(c):return RGBColor.from_string(c)
def box(sl,x,y,w,h,fill=PANEL,line=None,radius=False):
 sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill);sh.line.fill.background() if not line else None
 if line:sh.line.color.rgb=rgb(line)
 return sh
def text(sl,txt,x,y,w,h,size=24,color=WHITE,bold=False,font='Aptos',align=None):
 sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True;tf.margin_left=tf.margin_right=Inches(.015);tf.margin_top=tf.margin_bottom=0
 for i,line in enumerate(str(txt).split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name=font;p.font.size=Pt(size);p.font.color.rgb=rgb(color);p.font.bold=bold;p.space_after=Pt(8)
  if align is not None:p.alignment=align
 return sh
def pic(sl,name,x,y,w,h=None):
 p=ASSETS/name
 if h is None:return sl.shapes.add_picture(str(p),Inches(x),Inches(y),width=Inches(w))
 im=Image.open(p); iw,ih=im.size;scale=min(w/iw,h/ih);return sl.shapes.add_picture(str(p),Inches(x+(w-iw*scale)/2),Inches(y+(h-ih*scale)/2),width=Inches(iw*scale),height=Inches(ih*scale))
def base(title,kicker,source):
 sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=rgb(BG)
 idx=len(prs.slides);titles.append(title)
 box(sl,.55,.45,.34,.05,TEAL);text(sl,kicker.upper(),1.02,.34,10,.3,11,MUTED,True)
 text(sl,title,.6,.92,12.15,1.05,30,WHITE,True)
 box(sl,.6,6.96,12.1,.012,'294055');text(sl,source,.62,7.09,11.65,.22,8.5,MUTED);text(sl,f'{idx:02}',12.3,7.04,.4,.3,11,TEAL,True)
 return sl
def finish(sl,spoken,cue,transition,question,sources,scope,appendix=False):
 idx=len(prs.slides);speaker='Vishwanath Venugopal' if idx<=7 else 'Shaik Koduru Asif';dur=durations[idx-1] if idx<=14 else 0
 block=f'Slide {idx}: {titles[-1]}\nSpeaker: {speaker}\nDuration: {dur}s'+(' (appendix, not in main timing)' if appendix else '')+f'\n\nExact spoken script:\n{spoken}\n\nPointing cue: {cue}\nTransition: {transition}\nLikely question: {question}\n\nSource details:\n'+ '\n'.join(str(ROOT/p) for p in sources)+f'\nScope and numerical interpretation: {scope}\n'
 sl.notes_slide.notes_text_frame.text=block;notes.append(block)

def card(sl,x,y,w,num,label,accent=TEAL):
 box(sl,x,y,w,1.65,PANEL,radius=True);text(sl,num,x+.2,y+.15,w-.4,.67,37,accent,True);text(sl,label,x+.2,y+.93,w-.4,.55,16,MUTED)
def flow(sl,labels,y=2.7):
 w=(11.6-.24*(len(labels)-1))/len(labels)
 for i,l in enumerate(labels):
  x=.8+i*(w+.24);box(sl,x,y,w,1.1,PANEL,radius=True);text(sl,l,x+.15,y+.25,w-.3,.7,21,TEAL if i==len(labels)-1 else WHITE,True)
  if i<len(labels)-1:text(sl,'›',x+w+.03,y+.31,.2,.5,28,MUTED)

def native_chart(sl,cats,series,x,y,w,h,kind=XL_CHART_TYPE.BAR_CLUSTERED,maxval=1):
 data=CategoryChartData();data.categories=cats
 for n,vals,c in series:data.add_series(n,vals)
 chart=sl.shapes.add_chart(kind, Inches(x),Inches(y),Inches(w),Inches(h),data).chart
 chart.has_legend=len(series)>1
 if chart.has_legend:
  chart.legend.position=XL_LEGEND_POSITION.BOTTOM;chart.legend.font.size=Pt(12);chart.legend.font.color.rgb=rgb(MUTED)
 chart.chart_style=10
 chart.chart_format if hasattr(chart,'chart_format') else None
 chart.value_axis.minimum_scale=0;chart.value_axis.maximum_scale=maxval;chart.value_axis.tick_labels.font.size=Pt(11);chart.value_axis.tick_labels.font.color.rgb=rgb(MUTED)
 chart.category_axis.tick_labels.font.size=Pt(13);chart.category_axis.tick_labels.font.color.rgb=rgb(WHITE)
 for i,(_,vals,c) in enumerate(series):
  se=chart.series[i];se.format.fill.solid();se.format.fill.fore_color.rgb=rgb(c);se.format.line.color.rgb=rgb(c)
 chart.plots[0].has_data_labels=kind==XL_CHART_TYPE.BAR_CLUSTERED
 if chart.plots[0].has_data_labels:
  dl=chart.plots[0].data_labels;dl.font.size=Pt(12);dl.font.color.rgb=rgb(WHITE);dl.number_format='0.000'
 return chart

S='results/learning_v2_coverage_2026-10-25_v1/'
R='results/construction_change_regions_2026-10-26_v1/'
# 01
sl=base('Can AI distinguish real construction\nprogress from 3D scanning errors?','CV5570  /  Final presentation  /  02 November 2026','Source: MultiChange3D BikeParkingConstruction · pair 03 · genuine display samples')
pic(sl,'epoch_0.png',.5,2.15,6.1,2.85);pic(sl,'epoch_3.png',6.7,2.15,6.1,2.85)
text(sl,'EARLIER · EPOCH 0',.8,5.04,5,.3,12,ORANGE,True);text(sl,'LATER · EPOCH 3',6.95,5.04,5,.3,12,TEAL,True)
text(sl,'Robust 3D Change Detection for Construction Progress Monitoring',.8,5.65,11.7,.5,22,WHITE,True)
text(sl,'Vishwanath Venugopal · CE23B124     /     Shaik Koduru Asif · CE23B114',.8,6.25,11.7,.4,16,MUTED)
finish(sl,'These are two real construction scans from the same site, acquired at different epochs. The geometry changes, but so does the way it is observed. Our question is whether AI can distinguish physical construction progress from scanning errors. We present Robust 3D Change Detection for Construction Progress Monitoring for CV5570. I am Vishwanath, presenting the problem and experimental design; Asif will present the results, limitations, and practical workflow.','Point from earlier to later cloud.','First, consider what a scanner can make look like progress.','Are these genuine project scans?',['dataset/MultiChange3D/BikeParkingConstruction/pair_03/pair_03/pointCloud0.ply','dataset/MultiChange3D/BikeParkingConstruction/pair_03/pair_03/pointCloud1.ply'],'Display-only stride samples; no full-cloud prediction. Labels color candidate direction, not semantic components.')
# 02
sl=base('A difference in points is not always a change in work.','01 / The observation problem','Source: local pair 03 sample · explanatory perturbations, not benchmark measurements')
pic(sl,'scan_errors.png',.6,2.08,12.1,3.4)
text(sl,'Same surface. Different observations.',.8,5.7,11.6,.6,30,WHITE,True)
text(sl,'Displacement ≠ construction     •     Absence ≠ removal',.8,6.38,11.6,.35,19,MUTED)
finish(sl,'The grey points represent the original site sample. Cyan shows four deliberately altered observations. A small registration error displaces otherwise static surfaces. Noise scatters measurements around surfaces. Lower density increases gaps between nearest neighbours, while occlusion removes visibility altogether. These examples are explanatory transformations of our genuine scan, not the measured stress benchmark. The important distinction is that a detector sees geometry and sampling, not the construction process itself. Missing evidence can mean a blocked view rather than demolition, and a displaced surface need not mean that anything was installed. This motivates comparing several mechanisms under controlled conditions.','Sweep across the four panels, then underline absence versus removal.','We therefore need a bounded, testable research question.','How are illustrative and measured perturbations distinguished?',['dataset/MultiChange3D/BikeParkingConstruction/pair_03/pair_03/pointCloud1.ply','code/multipair_evaluate.py'],'Illustrative projection offsets/noise are not stress severity values; genuine cloud sampled for display.')
# 03
sl=base('Learn local evidence. Test what survives.', '02 / Research question and contributions','Sources: coverage design.json · interpretation_scope.json · original methods')
text(sl,'Can multiscale learned matching outperform\ngeometric distance on sampled construction cores?',.85,2.1,11.6,1.3,30,WHITE,True)
for x,n,l in [( .85,'01','Benchmark: added / removed / unchanged'),(4.85,'02','Robustness: simulate scanning errors'),(8.85,'03','Review: inspect errors and candidate regions')]:card(sl,x,4.05,3.6,n,l)
text(sl,'One site  /  same-site comparisons  /  human judgement remains essential',.85,6.18,11.7,.5,18,MUTED)
finish(sl,'We ask a narrower question than whether AI understands construction. Can learned local matching outperform geometric distance on sampled points from a construction scene, and what happens when the observations degrade? Our contributions are a directional, multi-epoch comparison, a controlled robustness analysis, and a review-oriented interpretation of predicted change. The final contribution is deliberately not automatic quantity takeoff. We distinguish a point prediction, a spatial candidate region, and a verified construction component. Those are different levels of evidence. The results support comparative same-site conclusions, with transparent caveats about the development history and the absence of an untouched final benchmark.','Point to the three contribution cards in order.','The dataset determines what can actually be measured.','What is new beyond a distance threshold?',[S+'design.json',S+'analysis/interpretation_scope.json','code/learning_v2_model.py'],'Prior held-out results informed follow-up design; avoid untouched benchmark or zero-leakage claims.')
#04
sl=base('Four epochs. Six pairs. Two directions.', '03 / Quantitative dataset','Source: BikeParkingConstruction · coverage design.json · fold_0/protocol.json')
for i in range(4):
 x=1+i*3.0;box(sl,x,2.25,2.2,.85,PANEL);text(sl,f'EPOCH {i}',x+.2,2.48,1.8,.4,22,TEAL,True)
text(sl,'01      02      03      12      13      23',1,3.58,11.1,.6,30,WHITE,True)
text(sl,'MultiChange3D · one construction site · point labels: changed / unchanged',1,4.17,11.2,.35,17,MUTED)
card(sl,1,4.65,5.35,'Earlier → later','Candidate removals on earlier points',ORANGE);card(sl,6.9,4.65,5.35,'Later → earlier','Candidate additions on later points',TEAL)
finish(sl,'The quantitative benchmark is BikeParkingConstruction from MultiChange3D: one site with four acquisition epochs. Every earlier–later combination gives six temporal pairs: zero–one, zero–two, zero–three, one–two, one–three, and two–three. We score both directions. Earlier points without corresponding later geometry are candidate removals. Later points without earlier counterparts are candidate additions. These remain geometric categories, not verified demolition or installation events. The six pairs share epochs and the five folds share a site, so the thirty pair-fold evaluations are not thirty independent studies. We use the benchmark labels for evaluation, while retaining ConSLAM only for qualitative context.','Trace the epoch sequence, then the two directional cards.','Next is the path from local evidence to review.','Are thirty evaluations independent?',[S+'design.json',S+'fold_0/protocol.json'],'One scene, four epochs, six overlapping pairs, 30 pair-fold evaluations, 60 scan directions; no independent-site claim from these means.')
#05
sl=base('From scans to decisions—with evidence attached.', '04 / End-to-end workflow','Sources: multipair_data.py · learning_v2_data.py · learning_v2_model.py · construction_change_regions.py')
flow(sl,['Paired scans','Buffered cores','Local evidence','Change scores','Review regions'])
text(sl,'Geometry baseline',1,4.3,3.4,.5,24,WHITE,True);text(sl,'Nearest distance\nSurface-normal projection',1,4.98,4,1.1,22,MUTED)
text(sl,'Learned comparison',5,4.3,3.6,.5,24,TEAL,True);text(sl,'Paired multiscale patches\n+ bounded geometry features',5,4.98,4.2,1.1,22,MUTED)
text(sl,'Output contract',9.35,4.3,3.2,.5,24,WHITE,True);text(sl,'Sampled point evidence\n—not a full-cloud result',9.35,4.98,3.1,1.2,21,MUTED)
finish(sl,'The workflow starts with paired scans and a spatially restricted set of original point cores. Each core receives local evidence from the two scans. The geometric branches use distances or local surface projection, whereas the learned branch combines multiscale patch information with bounded geometry features. The model produces a score, and a validation-chosen operating threshold turns that score into a point decision. A separate grouping step forms candidate review regions from positive sampled cores. Notice the output contract: we have not run or verified full-cloud inference here. A region is a grouping of sampled predictions, not a replacement for inspecting the underlying site geometry and its visibility.','Follow the five boxes left to right; point to the output contract.','The three methods interpret local evidence differently.','Which outputs are actually implemented?',['code/multipair_data.py','code/learning_v2_data.py','code/learning_v2_model.py','code/construction_change_regions.py'],'Scores and candidate grouping on sampled original held-out cores; no training or full-cloud inference performed by this build.')
#06 native mechanism diagram
sl=base('Three ways to interpret a missing neighbour.', '05 / Detection mechanisms','Sources: c2c_baseline.py · m3c2_baseline.py · learning_v2_model.py · coverage design.json')
for i,(label,col) in enumerate([('C2C',MUTED),('M3C2-style',PURPLE),('Multiscale V2',TEAL)]):
 x=.8+i*4.1;box(sl,x,2.12,3.85,4.35,PANEL,radius=True);text(sl,label,x+.2,2.35,3.4,.5,27,col,True)
 if i==0:
  for j in range(5):box(sl,x+.4+j*.52,3.32,.08,.08,GREY);box(sl,x+.55+j*.52,3.8,.08,.08,TEAL)
  text(sl,'↕ nearest distance',x+.4,4.15,3.1,.5,22,WHITE);text(sl,'Distance > threshold\nFixed reference: 0.075 m',x+.2,5.05,3.4,1,19,MUTED)
 elif i==1:
  box(sl,x+.35,3.35,3.1,.06,GREY);box(sl,x+.35,3.88,3.1,.06,TEAL);box(sl,x+1.7,3.0,.03,1.4,PURPLE)
  text(sl,'Project along a normal',x+.2,4.45,3.4,.5,21,WHITE);text(sl,'Local support + uncertainty\nUnknown when unsupported',x+.2,5.15,3.4,1,19,MUTED)
 else:
  for j,rad in enumerate([.62,.44,.25]):
   sh=sl.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x+1.85-rad),Inches(3.5-rad),Inches(2*rad),Inches(2*rad));sh.fill.background();sh.line.color.rgb=rgb(TEAL)
  text(sl,'0.15 / 0.40 / 0.80 m',x+.2,4.4,3.4,.5,22,WHITE,True);text(sl,'Shared patch encoder\nWeighted BCE—not focal loss',x+.2,5.15,3.4,1,19,MUTED)
finish(sl,'C2C asks how far the closest point is. Its fixed reference threshold is seven point five centimetres, while the calibrated version chooses its threshold on validation data. M3C2-style asks whether separation along an estimated surface normal exceeds a local uncertainty criterion. We call it style because this is our implementation, not a claim of reproducing every official M3C2 feature. It can return unknown when local support is insufficient. V2 instead compares paired patches at radii of fifteen, forty, and eighty centimetres. A shared point encoder pools local information, combines cross-scan interactions, and fuses a geometry branch. Crucially, the implemented objective is equal pair-and-present-class weighted binary cross entropy, not focal loss.','Point to each mechanism; emphasize the loss correction.','All comparisons depend on where cores are sampled.','Is V2 PointNet++ or a focal-loss model?',['code/c2c_baseline.py','code/m3c2_baseline.py','code/learning_v2_model.py',S+'design.json'],'Custom shared multiscale PointNet metric fusion; 64 points/patch, radii .15/.4/.8m, weighted BCE. Fixed C2C .075m is separate from calibrated baseline.')
#07 spatial protocol
sl=base('Spatial separation—not random point splitting.', '06 / Buffered evaluation','Source: coverage fold_0/protocol.json · design.json · interpretation_scope.json')
colors={'train':GREY,'validation':PURPLE,'test':TEAL};cells=protocol['cells']
for split,col in colors.items():
 for ix,iy in cells[split]:
  x=.9+ix*1.25;y=2.25+(2-iy)*1.2;box(sl,x,y,1.15,1.1,PANEL,line=col);box(sl,x+.36,y+.34,.43,.42,col)
for j,(s,c) in enumerate(colors.items()):text(sl,s.upper(),.9+j*1.75,6.12,1.7,.35,12,c,True)
card(sl,6.7,2.15,2.55,'8 m','Grid cells');card(sl,9.5,2.15,2.8,'2.5 m','Core boundary guard')
text(sl,'3,000 train  /  1,200 validation  /  1,500 test',6.7,4.22,5.6,.85,24,WHITE,True)
text(sl,'Counts per scan-direction, pair and fold\n90,000 clean test observations · 5 folds\n5 m cross-split guard; one site, reused validation',6.7,5.25,5.7,1.2,18,MUTED)
finish(sl,'This is the actual fold-zero cell assignment, drawn from the protocol manifest, not a five-stripe illustration. The grid uses eight-metre cells. Sampled cores are at least two point five metres from cell boundaries, giving a five-metre minimum cross-split core distance. The requested counts per scan-direction and fold are three thousand training, twelve hundred validation, and fifteen hundred test cores. Six pairs, two directions, and five folds therefore give ninety thousand clean held-out point observations. Evaluation sampling is uniform, not class-balanced. Validation chooses thresholds and variants. Earlier held-out results did influence this follow-up design, so we describe these as same-site comparisons and do not claim zero leakage or an untouched benchmark.','Point to inset core squares and the 2.5 m guard; then counts.','Asif will now show what this protocol reveals.','Why not random splitting, and is this untouched?',[S+'fold_0/protocol.json',S+'design.json',S+'analysis/interpretation_scope.json'],'Grid diagram uses actual cells, shown schematically. 1500*2*6*5=90000 clean held-out observations, repeated epochs not independent samples; requested training 180000 and validation72000 observations if all counts realized.')
#08 clean
sl=base('V2 improves clean comparison. Changed F1 stays modest.', '07 / Clean results · five spatial folds','Source: coverage analysis/clean_comparison.csv · 30 pair-fold evaluations · uniform sampled test cores')
methods=['previous_c2c_calibrated','previous_c2c_legacy_075','previous_m3c2_calibrated','previous_validation_selected','v2_validation_selected']
cats=['C2C calibrated','C2C fixed 0.075 m','M3C2 calibrated','Earlier selected','V2 selected']
native_chart(sl,cats,[('Binary macro F1',[float(metrics[m]['binary_macro_f1']) for m in methods],TEAL)],.65,2.0,7.0,4.7,maxval=.8)
card(sl,8.1,2.2,4.2,'0.654','Mean binary macro F1',TEAL);card(sl,8.1,4.12,4.2,'0.390','Mean directional changed F1',ORANGE)
text(sl,'+10.51 pp vs calibrated C2C\n+8.78 pp vs fixed C2C · F1, not accuracy',8.15,6.05,4.1,.6,17,MUTED)
finish(sl,'I am Asif, and these are the clean results across the thirty pair-fold evaluations. The selected V2 workflow reaches a mean binary macro F1 of zero point six five three seven. Calibrated C2C is zero point five four eight six, fixed C2C is zero point five six five nine, calibrated M3C2-style is zero point four one four two, and the earlier selected workflow is zero point six two one two. V2 improves by eight point seven eight percentage points over fixed C2C. But the orange number is equally important: mean directional changed F1 is only zero point three nine zero zero. The model improves comparison; it does not reliably solve every change case.','Read the teal bars, then explicitly point to orange changed F1.','Averaging choices explain why another F1 number looks higher.','Why is changed F1 much lower?',[S+'analysis/clean_comparison.csv'],'Exact V2 binary .6536591206208738; directional changed .3899887030904223. Calibrated C2C .5485634331059023; fixed .5658993214638721; calibrated M3C2 .414227689717096; earlier selected .6211906039032457. Unweighted same-site pair/fold/direction means, not accuracy.')
#09 aggregation
sl=base('One model. Different questions. Different F1s.', '08 / Metric honesty','Sources: class_support_and_pooled_confusions.csv · metric_ceiling_diagnostic.json')
card(sl,.85,2.15,3.7,'0.654','Mean binary macro F1',TEAL);card(sl,4.8,2.15,3.7,'0.390','Mean changed F1',ORANGE);card(sl,8.75,2.15,3.7,'0.741','Pooled 3-class macro F1',PURPLE)
text(sl,'21 of 60 directions contain no labelled change.',.95,4.38,11.6,.65,30,WHITE,True)
text(sl,'Absent-class F1 is set to zero.\nPerfect mean binary macro F1 is therefore 0.825—not 1.0.',.95,5.35,11.6,1.0,23,MUTED)
finish(sl,'These three numbers are not interchangeable. The first averages binary macro F1 across the spatial and directional evaluations. The second isolates the changed class under the same averaging rule. The third first pools out-of-fold confusion counts and then scores unchanged, removed, and added classes. That pooled three-class macro F1 is zero point seven four zero five. It weights observations differently, so it should not be presented as a replacement for the first two. Also, twenty-one of sixty evaluated directions contain no labelled change. Under the retained convention, an absent-class F1 is zero. Even perfect predictions therefore yield a mean binary macro F1 of zero point eight two five, not one.','Point across all three metrics; emphasize the different aggregation labels.','Next, we deliberately degrade the observations.','Is pooled F1 an accuracy claim or a directly comparable mean?',[S+'analysis/class_support_and_pooled_confusions.csv',S+'analysis/metric_ceiling_diagnostic.json',S+'analysis/clean_comparison.csv'],'Pooled three-class .7405075400072249; 39 positive-support and21 unchanged-only directions. Oracle ceiling diagnostic only; binary mean ceiling .825, directional changed mean ceiling .65.')
#10 stress
sl=base('Robust training helps under controlled misalignment.', '09 / Stress test · fold 0 only','Source: stress_condition_summary.csv · combined conditions · six pairs × three perturbation seeds')
conds=['combined_mild','combined_large'];ms=['previous_c2c_legacy_075','previous_c2c_calibrated','multiscale_robust_seed42']
series=[]
for m,n,c in zip(ms,['Fixed C2C','Calibrated C2C','Robust V2 seed 42'],[ORANGE,GREY,TEAL]):
 series.append((n,[float(next(r for r in stress if r['condition']==co and r['method']==m)['selection_binary_macro_f1_mean']) for co in conds],c))
native_chart(sl,['0.10 m + 1°','0.20 m + 2°'],series,.7,2.18,7.7,4.45,XL_CHART_TYPE.COLUMN_CLUSTERED,.8)
card(sl,8.75,2.3,3.65,'0.618','Severe combined · robust V2');card(sl,8.75,4.25,3.65,'0.418','Severe combined · fixed C2C',ORANGE)
text(sl,'Translation + rotation · no added noise\nCalibrated C2C: 0.572 · robust V2: 0.618\nFold 0 only; not selected five-fold workflow',8.8,6.0,3.65,.83,15,MUTED)
finish(sl,'The graph now changes scope: it uses fold-zero models only, evaluated on six pairs and three perturbation seeds per condition. We plot the actual mild and larger combined rigid conditions. Larger means a twenty-centimetre translation together with a two-degree rotation; it does not include an additional density or noise degradation. Robust V2 seed forty-two reaches zero point six one seven six, compared with zero point four one eight one for fixed C2C. The calibrated C2C result, zero point five seven two zero, is also shown to avoid overstating the gain against a weak operating point. This supports robustness of that trained variant, not an all-fold stress claim for the validation-selected workflow.','Point to severe group; compare all three bars, not just fixed C2C.','The mistakes still need to be visible.','Does this result cover all folds or a combined noise test?',[S+'analysis/stress_condition_summary.csv',S+'analysis/interpretation_scope.json','code/multipair_evaluate.py'],'Fold0 only,6 pairs x3 perturbation seeds. combined_large .2m translation+2degrees rotation only. Robust seed42 .6176271501676411 SD .0029596161322436405; fixed .41806009084589607 SD .038035885217122714; calibrated .5719863237438338.')
#11 uncertainty
sl=base('Expose the errors—not just the winning score.', '10 / Held-out uncertainty atlas','Source: construction_change_regions pair_03_added_cores.npz · all 7,500 later-epoch sampled out-of-fold cores')
pic(sl,'uncertainty_atlas.png',.65,2.04,12.05,3.9)
text(sl,'FP: review a false alarm     /     FN: recover missed evidence',.9,6.0,11.8,.5,23,WHITE,True)
text(sl,'Purple marks |score − 0.5| < 0.1, not a calibrated uncertainty probability.',.9,6.55,11.8,.3,15,MUTED)
finish(sl,'These panels use genuine held-out later-epoch cores from pair zero–three, rather than invented error examples. Grey is the sampled context. Orange highlights false alarms, where the model predicts change on an unchanged-labelled core. Cyan highlights missed change. Purple shows a simple score-boundary diagnostic: scores within zero point one of one half. That purple mask is not a calibrated uncertainty probability, and it is not the fold-specific decision threshold. The panels show why a review workflow must preserve both the predictions and the original evidence. We should not convert every positive point into progress, and we should not interpret a low score as proof that no work occurred.','Point to FP, then FN, then the explicitly labelled score diagnostic.','We can group evidence, but grouping does not create objects.','Are these actual errors, and are the probabilities calibrated?',[R+'pair_03_added_cores.npz',R+'source_predictions_manifest.json',R+'design.json'],'7500 pair03 later held-out cores across five folds. FP=(prediction1,label0);FN=(prediction0,label1); boundary diagnostic abs(model_score-.5)<.1. No posterior calibration claim.')
#12 regions
sl=base('191 candidate regions—not 191 construction objects.', '11 / From points to review candidates','Sources: candidate_regions.csv · pair_summary.csv · design.json · verification.json')
pic(sl,'region_centroids.png',.7,2.05,6.9,4.55)
card(sl,8,2.2,4.15,'191','Across six pairwise comparisons',TEAL);card(sl,8,4.15,4.15,'83','Only unchanged-labelled sampled members',ORANGE)
text(sl,'0.40 m grouping radius  /  ≥3 positive sampled cores',.9,6.5,11.6,.3,16,MUTED)
finish(sl,'We grouped predicted-positive held-out cores into spatial candidates using a fixed forty-centimetre connectivity radius and a minimum of three sampled positive members. Across six pairs, the saved output contains one hundred and ninety-one candidate regions. Eighty-three have only unchanged-labelled sampled members. That is a warning about false alarms, not an object-level accuracy statistic. The map shows the saved centroids for pair zero–three. Sampling gaps affect connectivity, and there is no verified component identity across pairs. These candidates can prioritize review; they cannot support installed-object counts, material quantities, or project completion percentages.','Point to the centroid map, then the 83 warning.','A separate field dataset illustrates why visibility matters.',[ 'What exactly does the count 83 mean?'][0],[R+'candidate_regions.csv',R+'pair_summary.csv',R+'design.json',R+'verification.json'],'191 CSV rows;83 rows with evaluation_changed_cores==0.90000 sampled observations. Fixed radius .4m,min3. No semantic object segmentation, object recall, verified quantities or temporal identity.')
#13
sl=base('ConSLAM is a reality check—not a labelled test.', '12 / Qualitative field context','Source: conslam_qualitative_2026-10-25_v2/inspection.json and conslam_qualitative.png')
pic(sl,'conslam_qualitative.png',.7,1.98,8.15,4.7)
text(sl,'4.84 seconds',9.05,2.5,3.4,.7,32,TEAL,True);text(sl,'Between selected mobile frames',9.05,3.27,3.2,.8,20,MUTED)
text(sl,'No official change labels.\nNo supervised V2 inference.\nNo F1 or IoU claim.',9.05,4.45,3.4,1.8,21,WHITE,True)
finish(sl,'ConSLAM provides qualitative field context, not another labelled construction-change test. The two selected mobile frames are approximately four point eight four seconds apart within one acquisition sequence. The saved inspection compares raw and pose-transformed geometry and shows the TLS reference separately. It explicitly states that the coordinate relationship between that TLS reference and the normalized mobile poses is unverified. There are no official per-point change labels for this comparison, so we do not report accuracy, F1, or IoU. We also do not claim supervised V2 inference on these scans. Its value is to illustrate viewpoint, coverage, and alignment issues that a practical review tool must handle.','Point to raw versus transformed context and the three no-claim lines.','We finish with what is implemented and what still needs development.','Was V2 quantitatively evaluated on ConSLAM?',['results/conslam_qualitative_2026-10-25_v2/inspection.json','results/conslam_qualitative_2026-10-25_v2/conslam_qualitative.png'],'4.839997s gap;12816/12935 selected-frame points. TLS104548932 points,50000 display sample. Pose/TLS relation unverified. No official labels, no learned inference demonstrated here.')
#14 conclusion practical demo
sl=base('Detect evidence. Review context. Verify progress.', '13 / Conclusion → live demonstration','Sources: clean_comparison.csv · construction candidate design.json · qualitative inspection.json')
box(sl,.85,2.12,5.55,2.62,PANEL,radius=True);text(sl,'IMPLEMENTED',1.1,2.38,4.8,.4,14,TEAL,True);text(sl,'Sampled change scores\nError inspection + candidate grouping',1.1,3.05,4.95,1.4,27,WHITE,True)
box(sl,6.8,2.12,5.55,2.62,PANEL,radius=True);text(sl,'FUTURE',7.05,2.38,4.8,.4,14,PURPLE,True);text(sl,'Verified BIM association\nQuantity / schedule integration',7.05,3.05,4.95,1.4,27,MUTED,True)
text(sl,'V2 improves same-site evidence. It does not certify installed work.',.95,5.15,11.55,.95,27,WHITE,True)
text(sl,'NEXT   5-minute offline evidence demonstration',.95,6.3,11.6,.4,20,TEAL,True)
finish(sl,'The takeaway is simple: detect evidence, review context, and only then verify progress. V2 improves our same-site sampled-core comparison, and robust training helps in the bounded stress setting. The implemented outputs are scores, inspectable errors, and candidate grouping. Verified BIM association, quantities, and schedule integration remain future work. We will now use the five-minute offline demonstration to inspect the evidence rather than claim automatic progress certification. After the demonstration, we reserve three minutes for questions.','Point to implemented versus future, then the demo transition.','Switch to the parent-owned offline demonstration.','Can this already produce installed quantities or EVM?',[S+'analysis/clean_comparison.csv',R+'design.json','results/conslam_qualitative_2026-10-25_v2/inspection.json'],'No BIM installation verification, quantities, EVM, or full-cloud inference claim. Main600s+demo300s+Q&A180s=1080s.')
#Appendix A
sl=base('Exact metrics and the absent-class ceiling.', 'Appendix A / Reference only','Sources: clean_comparison.csv · class_support_and_pooled_confusions.csv · metric_ceiling_diagnostic.json')
headers=['METHOD','BINARY MACRO','CHANGED F1','UNKNOWN']
for j,(x,w) in enumerate([( .9,4.8),(5.9,2),(8.3,2),(10.65,1.7)]):text(sl,headers[j],x,2.15,w,.4,13,MUTED,True)
for i,(m,label) in enumerate(zip(methods,cats)):
 y0=2.8+i*.54;box(sl,.85,y0-.08,11.55,.49,PANEL if i%2==0 else BG)
 vals=[label,f"{float(metrics[m]['binary_macro_f1']):.6f}",f"{float(metrics[m]['changed_f1']):.6f}",f"{100*float(metrics[m]['detector_unknown_rate']):.2f}%"]
 for j,x in enumerate([.95,5.9,8.3,10.65]):text(sl,vals[j],x,y0,4.7 if j==0 else 1.9,.35,18,TEAL if m=='v2_validation_selected' else WHITE)
text(sl,'Pooled 3-class macro: 0.740508  |  Oracle mean binary ceiling: 0.825',.95,5.95,11.6,.5,21,WHITE,True)
text(sl,'39 positive-support + 21 unchanged-only directions; zero convention retained.',.95,6.53,11.6,.3,16,MUTED)
finish(sl,'The exact table distinguishes calibrated and fixed baselines. The pooled score uses a different aggregation rule; the oracle ceiling is a diagnostic, not a predictive method.','Use the row and column labels before quoting a number.','Return to the relevant main result.','Why is the oracle ceiling below one?',[S+'analysis/clean_comparison.csv',S+'analysis/class_support_and_pooled_confusions.csv',S+'analysis/metric_ceiling_diagnostic.json'],'All exact values are read from CSV at build time. The upper diagnostic is absent-class dependent, not theoretical model performance.',True)
#Appendix B
sl=base('V2 is a custom metric-fusion network.', 'Appendix B / Architecture and selection','Sources: learning_v2_model.py · coverage design.json · interpretation_scope.json')
flow(sl,['3 paired scales','Shared 8→32→64','Max + mean','Interactions','Geometry fusion'],2.35)
text(sl,'Patch branch: 1,536 → 128     /     Geometry branch: 165 → 64',.95,4.0,11.6,.6,25,WHITE,True)
text(sl,'Concatenate 192 → 64 → 1 logit',.95,4.8,11.6,.5,25,TEAL,True)
text(sl,'Checkpoint: minimum clean-validation BCE\nThreshold + variant: clean-validation binary macro F1\nLoss: equal pair / present-class weighted BCE',.95,5.55,11.6,1.1,19,MUTED)
finish(sl,'The custom network shares an eight-channel point encoder, combines max and mean pooling, and forms cross-scan interactions. A one-hundred-and-sixty-five-feature geometry branch is fused with the patch branch. Checkpoints minimize clean-validation BCE; validation selects the threshold and variant. It is not a PointNet++ or KPConv reproduction.','Trace branch dimensions and selection rules.','Return to the clean-result slide.','How were checkpoints and thresholds selected?',['code/learning_v2_model.py',S+'design.json',S+'analysis/interpretation_scope.json'],'Radii .15/.4/.8m,64 points per patch;8→32→64 encoder;1536→128 patch head;165→64→64 geometry;192→64→1 head;dropout .1.Weighted BCE, not focal.',True)
#Appendix C
sl=base('Follow-ups are diagnostics—not extra headline wins.', 'Appendix C / Evidence boundaries','Sources: local follow-up manifests · external_open_space/metrics.csv · interpretation_scope.json')
text(sl,'Do not promote an exploratory improvement\nwithout validation and matching scope.',.95,2.2,11.5,1.25,31,WHITE,True)
for x,t,b in [(.95,'Local follow-ups','Intensity / temporal / fusion\nNot the headline test claim'),(5,'Open_space','Separate exposed indoor diagnostic\nC2C 0.860 / selected V2 0.786'),(9.05,'What is still missing','Multiple construction sites\nVerified component semantics')]:
 box(sl,x,4.12,3.45,2.1,PANEL,radius=True);text(sl,t,x+.18,4.37,3.1,.6,22,PURPLE,True);text(sl,b,x+.18,5.1,3.1,1.0,18,MUTED)
finish(sl,'The local follow-ups explore intensity, temporal evidence, and fusion, but they do not replace the headline selected workflow. Open_space is a separate already-exposed indoor diagnostic, not an untouched construction test. Calibrated C2C scored 0.8596 versus selected V2 at 0.7857. It is not pooled into construction-site means, and its data-identity caveat remains. The key missing evidence remains multi-site construction validation and verified component semantics.','Keep the three evidence scopes separate.','Return to conclusion.','Why not report a larger follow-up number?',[S+'analysis/interpretation_scope.json',S+'external_open_space/metrics.csv',R+'design.json'],'Exploratory follow-ups not promoted; external evaluation exists but no numeric headline here without method-by-method context.',True)

# Persist exact scripts and timing before export.
(OUT/'speaker_notes.md').write_text('# CV5570 — exact presentation script\n\nMain: 600 seconds. Appendix is not spoken during the timed talk.\n\n'+'\n---\n\n'.join(notes),encoding='utf-8')
timing=['# Presentation timing','', '| Slide | Speaker | Seconds | Cumulative |','|---|---|---:|---:|'];elapsed=0
for i,t in enumerate(durations,1):
 elapsed+=t;timing.append(f'| {i:02} — {titles[i-1]} | {"Vishwanath" if i<=7 else "Asif"} | {t} | {elapsed//60}:{elapsed%60:02} |')
timing += ['', 'Main slides: **600s (10:00)**. Vishwanath **300s**, Asif **300s**.', 'Live demonstration: **300s (5:00)**, parent-owned. Suggested split: Vishwanath first 150s, Asif final 150s.', 'Q&A: **180s (3:00)**. Both speakers answer in their evidence areas.', '**Total: 1080s (18:00)**.', '', 'Alternative if Q&A is separate: present the same **10:00 talk + 5:00 demo =15:00**; reserve the separate 3:00 Q&A outside the assessed presentation slot. Do not shorten the main script to eight minutes.', '', 'Rehearsal: durations include pointing, intentional pauses, and speaker handover. Appendix A–C have zero main-talk allocation. Spoken sentences are identical to embedded notes; source details and cues are not spoken.']
(OUT/'presentation_timing.md').write_text('\n'.join(timing),encoding='utf-8')
pptx=OUT/'CV5570_Final_Presentation.pptx';pdf=OUT/'CV5570_Final_Presentation.pdf';prs.save(pptx)
# Actual Microsoft PowerPoint PDF export; never synthesize a substitute PDF.
import pythoncom,win32com.client
pythoncom.CoInitialize();app=win32com.client.DispatchEx('PowerPoint.Application');app.Visible=True
try:
 deck=app.Presentations.Open(str(pptx),ReadOnly=True,Untitled=False,WithWindow=False)
 try:deck.SaveAs(str(pdf),32)
 finally:deck.Close()
finally:app.Quit();pythoncom.CoUninitialize()
# Render every exported PDF page and build contact sheets.
doc=fitz.open(pdf);pdftexts=[]
for i,page in enumerate(doc):
 page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(RENDERS/f'slide_{i+1:02}.png');pdftexts.append(page.get_text())
for start in range(0,len(doc),9):
 canvas=Image.new('RGB',(1500,930),'#172033');d=ImageDraw.Draw(canvas)
 for j in range(start,min(start+9,len(doc))):
  im=Image.open(RENDERS/f'slide_{j+1:02}.png');im.thumbnail((480,270));xx=10+(j-start)%3*500;yy=10+(j-start)//3*305
  canvas.paste(im,(xx,yy));d.text((xx+5,yy+275),f'{j+1:02} {titles[j][:52]}',fill='white')
 canvas.save(RENDERS/f'contact_sheet_{start//9+1:02}.png')
# Structural and export validations.
loaded=Presentation(pptx);bounds=[]
for i,sl in enumerate(loaded.slides,1):
 for sh in sl.shapes:
  if sh.left<0 or sh.top<0 or sh.left+sh.width>loaded.slide_width+200 or sh.top+sh.height>loaded.slide_height+200:bounds.append({'slide':i,'shape':sh.name})
assert len(doc)==17 and len(loaded.slides)==17
assert not bounds,bounds
assert all(sl.notes_slide.notes_text_frame.text==n for sl,n in zip(loaded.slides,notes))
assert len([x for x in RENDERS.glob('slide_*.png')])==17
assert '0.654' in pdftexts[7] and '0.390' in pdftexts[7]
assets=[]
for p in sorted(ASSETS.iterdir()):
 if p.is_file():assets.append(dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size))
manifest=dict(build='python-pptx authoring; Microsoft PowerPoint SaveAs PDF format 32; PyMuPDF rendering',source_root=str(ROOT),source_snapshot='Merged communication snapshot .publish\\Visioners_2.0_compact HEAD5477619; primary local arrays and CSV override unsupported narrative',sources=provenance,assets=assets)
(ASSETS/'asset_provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
validation=dict(passed=True,main_slides=14,appendix_slides=3,total_slides=17,aspect_ratio='16:9',main_seconds=600,speaker_main_seconds={'Vishwanath':300,'Asif':300},demo_seconds=300,qa_seconds=180,total_seconds=1080,all_pdf_slides_rendered=True,pdf_export_engine='Microsoft PowerPoint COM ExportAsFixedFormat(type=2)',pdf_pages=len(doc),editable_native_text=True,editable_metric_charts=2,editable_native_diagrams=True,embedded_notes_match_external=True,out_of_bounds=bounds,exact_metric_assertions_passed=True,region_count=191,unchanged_only_regions=83,clean_sampled_heldout_observations=90000,visual_inspection='Pending human/agent contact-sheet and selected-slide review after build',limitations=['Same-site comparison; earlier held-out evidence influenced follow-up design.','Fold0-only stress, not five-fold selected-workflow stress.','Sampled-core predictions only, not full-cloud inference.','ConSLAM qualitative, no labels and no supervised V2 inference.','No object counts, installation verification, quantities or EVM.','Runtime source path required for source rerun; copied rendered assets and chart values persist in PPTX.'],outputs={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in [pptx,pdf,OUT/'speaker_notes.md',OUT/'presentation_timing.md']})
(OUT/'presentation_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps({'complete':True,'slides':len(doc),'main_seconds':600,'output':str(OUT)},indent=2))
