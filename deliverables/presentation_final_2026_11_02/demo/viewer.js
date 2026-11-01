'use strict';
const $ = id => document.getElementById(id);
const canvas = $('scene'), ctx = canvas.getContext('2d');
let data, catalog, yaw = -.45, pitch = .75, zoom = 1, pan = [0,0], center = [0,0,0], extent = 1;
let hitBoxes = [], drag = null, serial = 0, selected = '', focusMode = false, pending = false;
const state = {renderCount:0, visible:0, loaded:false, errors:[], camera:[]};
window.demoState = state;
window.addEventListener('error', e => state.errors.push(e.message));
async function getJSON(url) { const r = await fetch(url, {cache:'no-store'}); if (!r.ok) throw Error(`${url}: HTTP ${r.status}`); return r.json(); }
function selectedScans() { return $('direction').value === 'both' ? [0,1] : [Number($('direction').value)]; }
function fit(rows) {
  const lo = [Infinity,Infinity,Infinity], hi = [-Infinity,-Infinity,-Infinity];
  rows.forEach(r => {for(let k=0;k<3;k++){lo[k]=Math.min(lo[k],r[k]);hi[k]=Math.max(hi[k],r[k]);}});
  if(!rows.length) return;
  center = lo.map((v,k)=>(v+hi[k])/2); extent = Math.max(...hi.map((v,k)=>v-lo[k]),1); zoom=1;pan=[0,0];
}
function project(p) {
  const x=p[0]-center[0], y=p[1]-center[1], z=p[2]-center[2];
  const a=Math.cos(yaw)*x-Math.sin(yaw)*y, b=Math.sin(yaw)*x+Math.cos(yaw)*y;
  const v=Math.cos(pitch)*b-Math.sin(pitch)*z, depth=Math.sin(pitch)*b+Math.cos(pitch)*z;
  const scale=Math.min(canvas.clientWidth,canvas.clientHeight)*.82/extent*zoom;
  return [canvas.clientWidth/2+a*scale+pan[0],canvas.clientHeight/2-v*scale+pan[1],depth];
}
function layerIndex(){return {gt:3,v2:4,c2c:5}[$('layer').value];}
function accepted(r) {
  const pred=r[layerIndex()], filter=$('filter').value;
  if(focusMode) return r[4]>=0 && r[4]!==r[3];
  if(filter==='changed')return pred===1;
  if(filter==='unchanged')return pred===0;
  if(filter==='unavailable')return pred<0;
  if(filter==='fp')return pred===1&&r[3]===0;
  if(filter==='fn')return pred===0&&r[3]===1;
  if(filter==='tp')return pred===1&&r[3]===1;
  return true;
}
function color(r,scan) {
  const p=r[layerIndex()];
  if(p<0)return '#6b7080';
  if($('layer').value!=='gt' && p!==r[3])return p===1?'#ff607d':'#c69aff';
  return p===1?(scan===0?'#ffb566':'#56d5a0'):'#8da3bd';
}
function regionScan(r){return r.kind==='candidate_removed'?0:1;}
function boxCorners(r){const lo=['x','y','z'].map(k=>Number(r['bbox_min_'+k])), hi=['x','y','z'].map(k=>Number(r['bbox_max_'+k]));return Array.from({length:8},(_,i)=>lo.map((v,k)=>(i&(1<<k))?hi[k]:v));}
function showRegion(value, move=false) {
  selected=value;$('region').value=value;
  const r=data.regions.find(r=>r.region_id===value);
  if(!r){$('details').textContent='Select a box in 3D or choose a candidate.';schedule();return;}
  $('details').textContent=`${r.region_id} · epoch ${r.cloud_epoch}\n${r.sampled_positive_cores} sampled positive cores\nMean score ${Number(r.mean_model_score).toFixed(3)} (not calibrated)\nGT changed / unchanged: ${r.evaluation_changed_cores} / ${r.evaluation_unchanged_cores}\nDescriptive point purity: ${(100*Number(r.evaluation_point_purity)).toFixed(1)}%\nSpatial folds: ${r.contributing_spatial_folds}\nFrozen thresholds: ${r.model_decision_thresholds}\nBBox XYZ metres:\n${['x','y','z'].map(k=>`${k}: ${Number(r['bbox_min_'+k]).toFixed(2)} … ${Number(r['bbox_max_'+k]).toFixed(2)}`).join('\n')}`;
  if(move){fit(boxCorners(r));extent=Math.max(extent*1.8,2);}
  schedule();
}
function schedule(){if(!pending){pending=true;requestAnimationFrame(()=>{pending=false;draw();});}}
function draw(){
  if(!data)return;
  const w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;
  if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
  ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
  const points=[], scans=selectedScans();let fp=0,fn=0,valid=0,total=0;
  for(const scan of scans)for(const r of data.scans[scan]){
    total++;valid+=r[6];if(r[4]===1&&r[3]===0)fp++;if(r[4]===0&&r[3]===1)fn++;
    if(accepted(r)){const p=project(r);points.push({p,r,scan});}
  }
  points.sort((a,b)=>a.p[2]-b.p[2]);
  const size=Number($('size').value);
  for(const {p,r,scan} of points){ctx.fillStyle=color(r,scan);ctx.globalAlpha=r[layerIndex()]===0?.58:.95;ctx.fillRect(p[0]-size/2,p[1]-size/2,size,size);}
  ctx.globalAlpha=1;hitBoxes=[];
  if($('boxes').checked && data.condition==='clean_seed42')for(const r of data.regions){
    if(!scans.includes(regionScan(r)))continue;
    const corners=boxCorners(r).map(project), chosen=r.region_id===selected;
    ctx.strokeStyle=chosen?'#fff6a1':(regionScan(r)?'#46aa8d':'#b58a5b');ctx.lineWidth=chosen?2.5:.8;
    ctx.beginPath();for(let i=0;i<8;i++)for(let k=0;k<3;k++){const j=i^(1<<k);if(i<j){ctx.moveTo(...corners[i].slice(0,2));ctx.lineTo(...corners[j].slice(0,2));}}ctx.stroke();
    const xs=corners.map(p=>p[0]),ys=corners.map(p=>p[1]);
    hitBoxes.push({id:r.region_id,x0:Math.min(...xs),x1:Math.max(...xs),y0:Math.min(...ys),y1:Math.max(...ys)});
    if(chosen){ctx.fillStyle='#fff6a1';ctx.font='12px Segoe UI';ctx.fillText(r.region_id,Math.min(...xs),Math.min(...ys)-6);}
  }
  // Fixed metric orientation triad rotates with the same 3D camera.
  const origin=project(center), labels=['X','Y','Z'],colors=['#ff7e8e','#6ad3a1','#71b9ff'];
  for(let k=0;k<3;k++){const p=center.slice();p[k]+=extent*.12;const q=project(p),dx=(q[0]-origin[0])/zoom,dy=(q[1]-origin[1])/zoom;ctx.strokeStyle=colors[k];ctx.fillStyle=colors[k];ctx.beginPath();ctx.moveTo(45,h-65);ctx.lineTo(45+dx,h-65+dy);ctx.stroke();ctx.fillText(labels[k],45+dx,h-65+dy);}
  $('count').textContent=points.length.toLocaleString();$('errors').textContent=`${fp} FP / ${fn} FN`;$('valid').textContent=`${valid.toLocaleString()} / ${total.toLocaleString()}`;
  state.visible=points.length;state.renderCount++;state.camera=[yaw,pitch,zoom,...pan,...center];state.selected=selected;state.focus=focusMode;
}
async function load(){
  const seq=++serial;state.loaded=false;$('status').textContent='Loading verified local assets…';
  try{
    const pair=$('pair').value,cond=$('condition').value;
    const next=await getJSON(`assets/${pair}_${cond}.json`);if(seq!==serial)return;data=next;
    focusMode=false;selected='';$('focus').textContent='FP / FN focus';$('filter').value='all';
    const clean=cond==='clean_seed42';$('boxes').disabled=!clean;$('region').disabled=!clean;
    $('region').replaceChildren(new Option(clean?'Choose a region…':'Clean candidates unavailable in stress',''));
    for(const r of data.regions)$('region').add(new Option(`${r.region_id} · ${r.sampled_positive_cores} cores`,r.region_id));
    $('region-note').textContent=clean?'Clean candidate boxes: original GT-free 0.4 m connected components, minimum 3 sampled positives. Fold-selected model may be a robust variant.':'Stress: saved degraded core coordinates, not a simulated animation. Clean component boxes are disabled because no saved stress grouping exists.';
    $('title').textContent=`Epoch ${pair[0]} → ${pair[1]} · ${clean?'clean':cond.replace('_seed42','')} · XYZ metres`;
    const s=data.support[0];
    $('support').textContent=`Actual saved support receipt (fold 0): ${s.original_source_points.toLocaleString()} earlier points; ${s.retained_target_points.toLocaleString()} retained later points. Display: ${data.scans.map(s=>s.length.toLocaleString()).join(' / ')} sampled cores (earlier / later). ${clean?'':'Stress exists here for fold 0 only (512 requested cores per epoch); other folds are NOT fabricated. Counts and errors are not a paired clean-versus-stress metric. Full support clouds are not displayed.'}`;
    $('model').textContent=Object.entries(catalog.models).filter(([f])=>clean||f==='0').map(([f,m])=>`Fold ${f}: ${m.model}, t=${m.threshold}`).join('\n');$('model').style.whiteSpace='pre-line';
    $('receipt').textContent='Each export verifies saved prediction SHA-256, frozen model hash and C2C ID / XYZ / GT alignment. Source receipts are retained locally.';
    fit(data.scans.flat());yaw=-.45;pitch=.75;showRegion('');state.loaded=true;state.pair=pair;state.condition=cond;$('status').textContent='Ready · saved evidence';schedule();
  }catch(e){$('status').textContent=`Failed: ${e.message}. Restart launcher to check assets.`;state.errors.push(String(e));}
}
for(const id of ['pair','condition'])$(id).addEventListener('change',load);
for(const id of ['direction','layer','filter','boxes','size'])$(id).addEventListener('input',()=>{focusMode=false;$('focus').textContent='FP / FN focus';schedule();});
$('region').addEventListener('change',()=>showRegion($('region').value,true));
$('reset').addEventListener('click',()=>{focusMode=false;$('focus').textContent='FP / FN focus';$('filter').value='all';fit(data.scans.flat());yaw=-.45;pitch=.75;schedule();});
$('focus').addEventListener('click',()=>{if(!data)return;$('layer').value='v2';$('filter').value='all';focusMode=!focusMode;$('focus').textContent=focusMode?'Exit FP / FN focus':'FP / FN focus';const rows=selectedScans().flatMap(s=>data.scans[s]).filter(r=>!focusMode||(r[4]>=0&&r[4]!==r[3]));fit(rows);schedule();});
canvas.addEventListener('contextmenu',e=>e.preventDefault());
canvas.addEventListener('pointerdown',e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,pan:e.shiftKey||e.button===2};});
canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(drag.pan){pan[0]+=dx;pan[1]+=dy;}else{yaw+=dx*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.008));}drag.x=e.clientX;drag.y=e.clientY;schedule();});
canvas.addEventListener('pointerup',e=>{if(drag&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<5){const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;const hit=hitBoxes.filter(b=>x>=b.x0&&x<=b.x1&&y>=b.y0&&y<=b.y1).sort((a,b)=>(a.x1-a.x0)*(a.y1-a.y0)-(b.x1-b.x0)*(b.y1-b.y0))[0];if(hit)showRegion(hit.id);}drag=null;});
canvas.addEventListener('pointercancel',()=>drag=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.25,Math.min(30,zoom*Math.exp(-e.deltaY*.001)));schedule();},{passive:false});
window.addEventListener('resize',schedule);
getJSON('assets/catalog.json').then(c=>{catalog=c;return load();}).catch(e=>{$('status').textContent=String(e);state.errors.push(String(e));});
