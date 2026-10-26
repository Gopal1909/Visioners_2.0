"""Self-contained offline demo: live synthetic geometry and saved real core results.

No original dataset, web service, GPU, training or full-cloud run required.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from m3c2_baseline import Config,estimate_normals,project_direction,classify


HTML='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>CV5570 measured 3D change demo</title><style>body{margin:0;background:#f8fafc;color:#172554;font:16px system-ui}main{max-width:1200px;margin:24px auto;padding:0 24px}h1{font-size:26px;margin-bottom:8px}select,button{padding:9px;margin:10px 16px 10px 0;font:inherit}canvas{width:100%;height:62vh;background:white;border:1px solid #cbd5e1;touch-action:none}p{max-width:1050px;line-height:1.5}#status{font-weight:600}.legend span{margin-right:24px}small{color:#475569}</style>
<main><h1>Robust 3D Change Detection for Construction Progress Monitoring</h1>
<p>Measured Bike Parking Construction results on the fixed spatial-test cores. Drag to rotate and use the wheel to zoom. These are saved experimental predictions. The command also runs a live synthetic geometry check.</p>
<label>Method <select id="method"><option value="gt">Ground truth</option><option value="c2c_prediction">C2C</option><option value="m3c2_prediction">M3C2-style</option><option value="learned_prediction">Siamese PointNet + NN</option></select></label>
<label>Epoch <select id="epoch"><option value="0">Scan 0 removals</option><option value="1">Scan 1 additions</option></select></label>
<label><input id="common" type="checkbox" checked> Same M3C2-valid support</label><button id="reset">Reset view</button>
<p class="legend"><span style="color:#2563eb">● Unchanged</span><span style="color:#dc2626">● Added or removed candidate</span><span style="color:#9ca3af">● Unknown or excluded</span></p>
<canvas id="view" width="1400" height="850"></canvas><p id="status"></p>
<small>Common-support clean macro F1: C2C 0.4118, M3C2-style 0.2342, learned 0.3817. Coverage 66.875%. Same-scene spatial holdout; historical C2C threshold and exploratory pilot used this pair. Geometric changes do not establish structural completion or schedule progress.</small></main>
<script>const D=__DATA__;const c=document.getElementById('view'),x=c.getContext('2d');let az=-.55,el=.65,zoom=1,drag=false,last;
function draw(){const e=document.getElementById('epoch').value,m=document.getElementById('method').value,common=document.getElementById('common').checked;const p=D['xyz'+e],v=D[m+e],valid=D['m3c2_prediction'+e];const mid=[0,0,0];for(let j=0;j<3;j++)mid[j]=(Math.min(...p.map(a=>a[j]))+Math.max(...p.map(a=>a[j])))/2;const a=Math.cos(az),b=Math.sin(az),d=Math.cos(el),f=Math.sin(el);const rows=p.map((q,i)=>{let X=q[0]-mid[0],Y=q[1]-mid[1],Z=q[2]-mid[2];let xx=a*X-b*Y,yy=b*X+a*Y;return{i,xx,yy:d*yy-f*Z,z:f*yy+d*Z}}).sort((a,b)=>a.z-b.z);const span=Math.max(...rows.map(a=>Math.abs(a.xx)),...rows.map(a=>Math.abs(a.yy)),1),s=Math.min(c.width,c.height)*.43/span*zoom;x.clearRect(0,0,c.width,c.height);let n=0;for(const r of rows){let q=common&&valid[r.i]<0?-1:v[r.i];if(q<0)n++;x.fillStyle=q<0?'#9ca3af':q===1?'#dc2626':'#2563eb';x.globalAlpha=q<0?.3:.85;x.beginPath();x.arc(c.width/2+r.xx*s,c.height/2-r.yy*s,2.3,0,7);x.fill()}x.globalAlpha=1;document.getElementById('status').textContent=p.length+' original cores. '+n+' gray unknown/excluded points. Red is a candidate geometric or learned change.'}
c.onpointerdown=e=>{drag=true;last=[e.clientX,e.clientY];c.setPointerCapture(e.pointerId)};c.onpointerup=()=>drag=false;c.onpointermove=e=>{if(drag){az+=(e.clientX-last[0])*.008;el+=(e.clientY-last[1])*.008;last=[e.clientX,e.clientY];draw()}};c.onwheel=e=>{e.preventDefault();zoom=Math.max(.3,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw()};document.querySelectorAll('select,input').forEach(e=>e.onchange=draw);document.getElementById('reset').onclick=()=>{az=-.55;el=.65;zoom=1;draw()};draw();</script></html>'''


def run(bundle,out):
    bundle=Path(bundle);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (bundle/'checksums.json').exists():
        for name,sha in json.loads((bundle/'checksums.json').read_text()).items():
            assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==sha,name
    # A real cylinder/PCA computation, not an archive-only smoke check.
    u,v=np.meshgrid(np.arange(-.5,.501,.02),np.arange(-.5,.501,.02))
    p=np.column_stack([u.ravel(),v.ravel(),np.zeros(u.size)])
    q=p+[0,0,.12];centers=np.zeros((1,3));cfg=Config(projection_radii_m=(.1,))
    t0,t1=cKDTree(p),cKDTree(q);n=estimate_normals(t0,centers,cfg)
    r=project_direction(t0,t1,centers,n,cfg);d=classify(r,0,.02)
    np.testing.assert_allclose(r['signed_distance_m'],.12,atol=1e-10)
    assert d['prediction'][0,0]==1
    with np.load(bundle/'demo_measurements.npz') as z:
        data={k:(np.round(z[k],5).tolist() if k.startswith('xyz') else z[k].tolist())
              for k in z.files if k.startswith(('xyz','gt','c2c_prediction','m3c2_prediction','learned_prediction'))}
    path=out/'change_demo.html';path.write_text(HTML.replace('__DATA__',json.dumps(data)),encoding='utf-8')
    report=dict(synthetic_offset_expected_m=.12,synthetic_offset_recovered_m=float(r['signed_distance_m'][0,0]),
        synthetic_passed=True,measured_scan0_cores=len(data['xyz0']),measured_scan1_cores=len(data['xyz1']),
        real_results='saved measured spatial-test predictions; not a live dataset rerun',html=str(path.resolve()))
    (out/'demo_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    default=root/'artifacts' if (root/'artifacts').exists() else root/'final_submission/artifacts'
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',default=str(default))
    p.add_argument('--output',default=str(default.parent/'demo'))
    a=p.parse_args();run(a.bundle,a.output)
