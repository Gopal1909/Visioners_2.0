import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const runtime='C:/Users/prani/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=runtime+'/node/node_modules';
const skill='C:/Users/prani/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations';
const {Presentation,PresentationFile}=await import(pathToFileURL(runtime+'/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs').href);
const {finalizePresentation,applyPresentationChartFont,resolvePresentationFont}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const workspace=process.cwd();
const finalDir=await fs.access(path.join(workspace,'artifacts')).then(()=>workspace).catch(()=>path.join(workspace,'final_submission'));
const art=path.join(finalDir,'artifacts');
const build=path.join(workspace,'results/submission_build/slides');
await fs.mkdir(build,{recursive:true});
const readJson=async n=>JSON.parse(await fs.readFile(path.join(art,n),'utf8'));
const data=await readJson('submission_data.json'),metrics=await readJson('presentation_metrics.json'),points=await readJson('spatial_points.json');
const supp=await readJson('submission_data.json');
const trainLines=(await fs.readFile(path.join(art,'learned/training_history.csv'),'utf8')).trim().split(/\r?\n/);
const history=trainLines.slice(1).map(l=>{let x=l.split(',').map(Number);return {epoch:x[0],train:x[1],cal:x[2]}});
const FONT=resolvePresentationFont({fontFamily:'Arial'});
const roundChart=n=>Number(Number(n).toFixed(6));
const p=Presentation.create({slideSize:{width:1280,height:720}});
const C={C2C:'#2563eb',M3C2:'#ea580c',Learned:'#0f766e'};
const methodNames=['C2C common','M3C2-style','Siamese PointNet common'];
const notes=[];
function txt(s,text,x,y,w,h,size=28,bold=false,color='#0f172a'){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=text;t.text.style={typeface:FONT,fontSize:size,bold,color,insets:0,verticalAlignment:'top',autoFit:'none'};return t;
}
function slide(title,speaker,seconds,note,sources){
 const s=p.slides.add();s.background.fill='#ffffff';
 if(title)txt(s,title,64,42,1150,85,44,true);
 txt(s,'CV5570   2 November 2026',64,682,700,22,16,false,'#64748b');
 txt(s,String(p.slides.items.length),1180,682,35,22,16,false,'#64748b');
 s.speakerNotes.text=`${speaker}; ${seconds} seconds.\n${note}\nChart values rounded to six decimals for portable editable Excel workbooks; full precision remains in CSV evidence.\n\nSources: ${sources}`;
 notes.push({slide:p.slides.items.length,title,speaker,seconds,note,sources});return s;
}
async function image(s,file,x,y,w,h){let b=await fs.readFile(path.join(art,'figures',file));s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt:file,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function table(s,values,x,y,width,height,widths){
 const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width,height,columnWidths:widths,values});
 t.borders.assign({style:'solid',fill:'#cbd5e1',width:1});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){
  const cell=t.getCell(r,c);cell.fill=r===0?'#e2e8f0':(r%2?'#ffffff':'#f8fafc');
  cell.text.style={typeface:FONT,fontSize:25,bold:r===0,color:'#0f172a',verticalAlignment:'middle',insets:12};
 }
 return t;
}
function lineChart(s,title,x,y,w,h,xValues,series,xTitle,maxX,maxY=.7,minX=0){
 const chart=s.charts.add('scatter',{
  position:{left:x,top:y,width:w,height:h},title,titleTextStyle:{fontSize:28,fill:'#0f172a',bold:true},
  series:series.map((v,i)=>({name:v.name,values:v.values.map(roundChart),xValues:xValues.map(roundChart),fill:v.color,line:{fill:v.color,width:3},marker:{symbol:'circle',size:7,fill:v.color}})),
  scatterOptions:{style:'lineWithMarkers'},hasLegend:false,
  xAxis:{min:minX,max:maxX,title:{text:xTitle,textStyle:{fontSize:22}},textStyle:{fontSize:20,fill:'#334155'},numberFormatCode:'0.##',majorGridlines:null},
  yAxis:{min:0,max:maxY,title:{text:'Macro F1',textStyle:{fontSize:22}},textStyle:{fontSize:20,fill:'#334155'},numberFormatCode:'0.0',majorGridlines:{fill:'#e2e8f0',width:1}},
  chartFill:'#ffffff',plotAreaFill:'#ffffff',chartLine:{fill:'none',width:0},
 });applyPresentationChartFont(chart,{fontFamily:FONT});return chart;
}
function legend(s,y=590,learned=true){txt(s,'C2C',90,y,220,36,24,true,C.C2C);txt(s,'M3C2-style',390,y,300,36,24,true,C.M3C2);if(learned)txt(s,'Siamese PointNet + NN',760,y,440,36,24,true,C.Learned);}
function curves(family){
 let severity=[...new Set(metrics.filter(r=>r.family===family).map(r=>r.severity))].sort((a,b)=>a-b);
 let x=[family==='sparsity'?100:0,...severity.map(v=>v*(family==='translation'||family==='sparsity'||family==='noise'?100:1))];
 const series=methodNames.map((m,i)=>({name:['C2C','M3C2-style','Siamese PointNet + NN'][i],color:Object.values(C)[i],values:[metrics.find(r=>r.method===m&&r.condition==='clean').combined_macro_f1,...severity.map(v=>metrics.find(r=>r.method===m&&r.family===family&&r.severity===v).combined_macro_f1)]}));
 const order=x.map((_,i)=>i).sort((a,b)=>x[a]-x[b]);
 return {x:order.map(i=>x[i]),series:series.map(s=>({...s,values:order.map(i=>s.values[i])}))};
}

// 1: real measured site geometry, never a decorative stock image.
let s=slide('', 'Vishwanath Venugopal',20,
 'Our project compares geometric and learned ways to locate additions and removals in repeated construction scans. We focus on controlled robustness, fair evaluation and explicit unsupported areas. The picture shows actual labelled original cores from Bike Parking Construction. We make no claim that a geometric discrepancy proves progress or structural completion.',
 'User proposal Robust_3D_Change_Detection_v2.pdf; artifacts/historical_pilot/primary_comparison.csv; https://3dom-fbk.github.io/multichange3d/');
txt(s,'Robust 3D\nChange Detection\nfor Construction\nProgress Monitoring',64,65,670,330,56,true);
txt(s,'Vishwanath Venugopal  CE23B124\nShaik Koduru Asif  CE23B114',68,455,650,92,25);
txt(s,'Computer Vision for Construction Engineering',68,580,1050,44,25,false,'#475569');
await image(s,'site_overview.png',680,165,550,390);

s=slide('Data and directional labels','Vishwanath Venugopal',35,
 'All quantitative results use MultiChange3D Bike Parking Construction pair 01, about 4.44 and 5.09 million points. The binary field is unchanged zero or changed one in each epoch. A changed earlier point means removed, and a changed later point means added. We verified the official label preparation and classical binary metric code. ConSLAM has no verified per-point change labels here. Its scans and poses provide qualitative context only. Additional official download links redirected to sign-in, so this submission contains no cross-pair or cross-scene experiment.',
 'artifacts/dataset_manifest.json; artifacts/data_sources_and_access.md; https://github.com/3DOM-FBK/multichange3d/blob/main/scripts/prepare_gt_labels.py');
table(s,[['Source epoch','Points','cd_type = 1 means'],['Scan 0 earlier','4,444,935','Removed'],['Scan 1 later','5,089,793','Added']],90,180,1100,240,[360,300,440]);
txt(s,'cd_type = 0 means unchanged in both epochs',92,460,1100,46,30,true);
txt(s,'Quantitative: MultiChange3D pair_01\nQualitative: local ConSLAM Sequence 3 only',92,535,1100,90,27);

s=slide('Three detectors with different assumptions','Vishwanath Venugopal',65,
 'C2C uses Euclidean nearest-neighbour distance above 7.5 centimetres, fixed from the historical same-pair sweep. M3C2-style estimates source normals at 10, 20 and 40 centimetre radii, then compares two projected neighbourhood means in the same 10 centimetre radius cylinder of half-length one metre. At least ten projected points per epoch and stable normals are required; otherwise it abstains. We checked the population spread and registration-term placement against CloudCompare v2.13.2 source, but did not validate output equivalence. Our learned baseline shares a point MLP between paired 128-point patches, pools geometry and combines it with an NN prior. Ground-truth labels enter supervision only.',
 'artifacts/geometry_frozen.json; artifacts/learned/model_config.json; artifacts/historical_pilot/reference_conventions.json; https://arxiv.org/abs/1302.1183; https://arxiv.org/abs/1612.00593');
txt(s,'C2C',72,165,500,42,32,true,C.C2C);txt(s,'Nearest opposite-epoch point\nChanged when distance > 7.5 cm',72,220,520,100,28);
txt(s,'M3C2-style',72,355,520,42,32,true,C.M3C2);txt(s,'Source PCA normals and shared cylinders\nProjection radius 10 cm, half-length 1 m\nLoD95 and 2 cm minimum-effect floor\nInvalid support produces unknown',72,410,555,165,26);
txt(s,'Siamese PointNet + NN',690,165,520,42,32,true,C.Learned);
txt(s,'Shared 3–32–64 point encoder\nMasked max and mean pooling\n128 neighbours within 0.60 m\nPaired geometry plus proximity prior\nWeighted binary supervision',690,225,510,250,28);
txt(s,'Registration uncertainty is separate and set to zero\nLoD is conditional on that assumption',690,510,510,100,24,false,'#475569');

s=slide('Buffered spatial evaluation','Vishwanath Venugopal',55,
 'We define these X regions using geometry, with 2.22 metre gaps. The gap exceeds twice the largest clean geometric support halo, so clean train, calibration and test patches do not share neighbourhood support. Both acquisition epochs occur in every split, which means this is a same-scene spatial holdout for the learned model, not cross-scene generalization. We sample original indices across occupied two-metre tiles and use complete clouds for geometric support. Sparse tiles therefore get relatively more weight. Calibration diagnostics freeze geometry before new test and stress scoring. Historical full-pair C2C threshold selection and the earlier exploratory pilot remain disclosed sources of test exposure.',
 'artifacts/split_config.json; artifacts/geometry_frozen.json; artifacts/spatial_points.json; 220 displayed original Scan 0 cores per split');
const scatter=s.charts.add('scatter',{position:{left:65,top:150,width:700,height:445},series:Object.entries(points).map(([name,v],i)=>({name,values:v.map(z=>roundChart(z[1])),xValues:v.map(z=>roundChart(z[0])),fill:['#2563eb','#ca8a04','#0f766e'][i],marker:{symbol:'circle',size:4,fill:['#2563eb','#ca8a04','#0f766e'][i]},line:{fill:'none',width:0}})),scatterOptions:{style:'marker'},hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:21}},xAxis:{title:{text:'X (m)',textStyle:{fontSize:22}},min:27,max:56,textStyle:{fontSize:20}},yAxis:{title:{text:'Y (m)',textStyle:{fontSize:22}},min:0,max:21,textStyle:{fontSize:20},majorGridlines:{fill:'#e2e8f0',width:1}}});applyPresentationChartFont(scatter,{fontFamily:FONT});
table(s,[['Region','Cores per epoch'],['Train','8,000'],['Calibration','1,000'],['Test','4,000']],800,180,410,265,[210,200]);
txt(s,'2.22 m buffers',805,475,410,50,34,true);txt(s,'Same scene and epochs\nNo cross-scene claim',805,545,410,90,26);

s=slide('C2C leads on this clean spatial region','Vishwanath Venugopal',65,
 'All three methods use the exact same valid original indices here: 5,350 of the requested 8,000 directional observations. C2C macro F1 is 0.4118, the learned baseline 0.3817, and M3C2-style 0.2342. M3C2 abstains on 33.125 percent. C2C and learned predictions have no detector abstentions; they are masked here for fairness, and their all-core scores are retained separately. These are lower than the earlier whole-scene pilot because the spatial region and sampled class mix differ. Only 27 removal and 86 addition labels remain after validity filtering, so recall is strongly conditional. We preserve the negative finding rather than promote a more favorable sweep row.',
 'artifacts/primary_comparison.csv; artifacts/all_spatial_test_metrics.csv; historical full-pair values are in a separate scope in artifacts/all_evaluated_results.csv');
const ordered=['C2C common','Siamese PointNet common','M3C2-style'].map(m=>data.primary.find(r=>r.method===m));
table(s,[['Method','Macro F1','Macro mIoU','Detector unknown'],...ordered.map((r,i)=>[["C2C","Siamese + NN","M3C2-style"][i],r.combined_macro_f1.toFixed(4),r.combined_macro_mIoU.toFixed(4),(100*r.unknown_rate).toFixed(3)+'%'])],90,180,1100,260,[380,220,230,270]);
txt(s,'5,350 valid cores out of 8,000 requested',95,485,1100,55,34,true);
txt(s,'Common evaluation coverage 66.875%\nC2C and learned all-core results stay separately labelled',95,565,1100,85,26);

s=slide('Registration errors change each detector differently','Shaik Koduru Asif',50,
 'These are one-sided simulated shifts and rotations of Scan 1, with exact Stage 3 seed and directions. All method settings stay fixed. The smaller perturbation can improve scores for C2C or the learned network in this particular direction, so there is no monotonic or universal benefit from misalignment. At 20 centimetres, the longitudinal comparison on identical clean/current valid cores gives macro-F1 changes of about minus 0.105 for C2C, minus 0.139 for M3C2 and minus 0.045 for the learned network. Each severity uses a saved original-index mapping. The report gives both condition-specific scores and matched clean deltas.',
 'artifacts/presentation_metrics.json; artifacts/matched_clean_deltas.csv; artifacts/perturbation_geometry.json');
let a=curves('translation'),b=curves('rotation');lineChart(s,'Translation',60,155,560,385,a.x,a.series,'Shift magnitude (cm)',20);lineChart(s,'Rotation',660,155,560,385,b.x,b.series,'Angle (degrees)',1);legend(s,555);
txt(s,'Fixed parameters and identical perturbations across methods',70,625,1130,38,24,false,'#475569');

s=slide('Density loss and noise also change valid support','Shaik Koduru Asif',50,
 'Sparsity retains exactly the same seeded subset for every method, and evaluation keeps only retained Scan 1 cores. Noise is independent Gaussian displacement per coordinate. The model was trained on clean data only. M3C2 coverage falls sharply under severe noise, so a valid-only score can hide difficult abstentions. Our matched-clean table intersects clean and current validity to separate some support effects from actual prediction changes. On 10 percent retention, the learned model loses roughly 0.096 macro F1 on that longitudinal support, while the corresponding C2C change is around minus 0.005. Those observations are single realizations with changing class counts, not a universal robustness ordering.',
 'artifacts/matched_condition_metrics.csv; artifacts/matched_clean_deltas.csv; artifacts/coverage.png');
a=curves('sparsity');b=curves('noise');lineChart(s,'Point retention',60,155,560,385,a.x,a.series,'Scan 1 retained (%)',100);lineChart(s,'Coordinate noise',660,155,560,385,b.x,b.series,'Per-axis SD (cm)',7.5);legend(s,555);txt(s,'Condition-specific common-valid scores; coverage is reported separately',70,625,1140,38,24,false,'#475569');

s=slide('Occlusion requires a visibility-aware interpretation','Shaik Koduru Asif',40,
 'The main spatial-test stripe happens to be outside both planned deletion cylinders. Its unchanged scores cannot establish occlusion robustness. We therefore add a separately labelled geometric supplement on the original whole-scene pilot indices. Strict protocol A retains all requested earlier-scan cores. Protocol B excludes cores inside the known synthetic blind cylinder; it changes evaluation support, not the detector. At radius 7.6 metres, M3C2 coverage is about 52 percent in A and 67 percent in B. Missing counterparts remain unknown, not confirmed physical removals. We do not evaluate the learned model on training-region supplement cores and call it a held-out result.',
 'artifacts/occlusion_supplement/geometry_metrics.csv; artifacts/occlusion_supplement/geometry_frozen.json; all-scene geometric supplement scope');
for(let j=0;j<2;j++){let protocol=['A','B'][j],x=[0,4.6,7.6],series=['C2C common','M3C2-style'].map((m,i)=>({name:i?'M3C2-style':'C2C',color:i?C.M3C2:C.C2C,values:x.map(v=>data.supplementary_occlusion.find(r=>r.method===m&&r.condition===(v===0?'clean':`occlusion_${v}`)&&(v===0||r.protocol===protocol)).combined_macro_f1)}));lineChart(s,protocol==='A'?'A: strict scoring':'B: blind cylinder excluded',j?660:60,175,560,375,x,series,'Deleted cylinder radius (m)',7.6,.8)}legend(s,565,false);txt(s,'Separate all-scene geometric supplement; not the learned spatial holdout',70,625,1140,36,24,false,'#475569');

s=slide('The learned baseline shows a large spatial gap','Shaik Koduru Asif',35,
 'We trained one small architecture for 30 predetermined epochs. Calibration loss, not test performance, selected the first epoch, and calibration F1 selected the 0.7 threshold. Training loss declines while calibration loss worsens. Calibration macro F1 is about 0.765 versus 0.394 on spatial test when both use all requested cores. The three-method common-support test score is separately 0.382. The model never receives change labels as input or absolute coordinates, but a single-scene split still shifts geometry and class balance. This compact baseline does not reproduce Siamese KPConv or characterize all learning methods.',
 'artifacts/learned/training_history.csv; artifacts/learned/model_frozen.json; artifacts/learned/model_config.json; artifacts/primary_comparison.csv');
const trainChart=s.charts.add('line',{position:{left:60,top:170,width:730,height:415},categories:history.map(r=>String(r.epoch)),series:[{name:'Training BCE',values:history.map(r=>roundChart(r.train)),line:{fill:C.C2C,width:3}},{name:'Calibration BCE',values:history.map(r=>roundChart(r.cal)),line:{fill:C.M3C2,width:3}}],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:22}},xAxis:{title:{text:'Epoch',textStyle:{fontSize:22}},textStyle:{fontSize:17}},yAxis:{min:0,max:.9,title:{text:'Weighted BCE',textStyle:{fontSize:22}},textStyle:{fontSize:20},majorGridlines:{fill:'#e2e8f0',width:1}}});applyPresentationChartFont(trainChart,{fontFamily:FONT});
table(s,[['All-core scope','Macro F1'],['Calibration','0.7649'],['Spatial test',data.learned_spatial_test_all_macro_f1.toFixed(4)]],825,210,385,215,[220,165]);
txt(s,'Epoch 1 selected\nThreshold 0.7 frozen',830,480,380,90,28,true);
txt(s,'No test-driven checkpoint selection or stress retuning',70,625,1130,38,24,false,'#475569');

s=slide('ConSLAM provides qualitative site context','Shaik Koduru Asif',35,
 'We reuse local Sequence 3 LiDAR, normalized poses and a bounded TLS sample. Only 520 available pose timestamps match the 6,014 LiDAR files exactly. We show the first two matched frames, 4.84 seconds apart, using the supplied poses. Median sampled proximity falls from 0.395 metres before pose application to 0.081 afterwards. That is neither a registration RMS nor change-detection accuracy. The frames belong to one acquisition sequence, and we did not verify a common frame against TLS. Therefore this supports qualitative discussion of alignment and coverage, not a monthly construction-change or progress claim.',
 'artifacts/conslam/inspection.json; https://www.repository.cam.ac.uk/items/693c260b-e145-4f01-ad88-d2031e4c0590; https://github.com/mac137/ConSLAM');
await image(s,'paper_conslam.png',100,165,1080,355);
txt(s,'Two LiDAR frames, 4.84 s apart, with supplied poses',90,545,1120,50,30,true);
txt(s,'No verified change labels, accuracy score or cross-month progress claim',90,610,1120,42,25,false,'#475569');

s=slide('Findings and limits','Shaik Koduru Asif',30,
 'C2C remains the strongest clean method on the matched spatial region. M3C2-style passes geometry tests but has many false positives and unknowns. The compact learned model is feasible on the laptop but exhibits a large spatial validation gap. Our strongest contribution is fair support-aware robustness analysis and an auditable negative result. Limitations include one scene and pair, one seed, historical threshold exposure, uncalibrated registration uncertainty and no independent M3C2 reference-output comparison. The offline demo runs live synthetic geometry and lets you rotate saved measured predictions. Geometry localizes candidate changes; semantic and BIM evidence would be needed to infer completed work or schedule status.',
 'artifacts/submission_data.json; report/project_report.pdf; README.md; demo uses saved real measurements plus live synthetic check');
txt(s,'C2C leads on the matched clean region',75,175,1130,55,34,true);
txt(s,'M3C2 validity and learned spatial shift limit reliability',75,260,1130,70,31);
txt(s,'One scene and pair, one seed, conditional uncertainty\nCandidate geometry alone does not certify construction progress',75,365,1130,130,29);
txt(s,'Offline demo:  python code/demo.py',75,560,1130,60,31,true,'#2563eb');

await fs.writeFile(path.join(build,'slides_notes.json'),JSON.stringify(notes,null,2));
await fs.writeFile(path.join(finalDir,'presentation/speaker_notes.md'),notes.map(n=>`## ${n.slide} ${n.title||'Title'}\n\n${n.speaker}, ${n.seconds} seconds\n\n${n.note}\n\nSources: ${n.sources}\n`).join('\n'));
const candidate=path.join(build,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
for(let i=0;i<p.slides.items.length;i++){
 let b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
const finalPath=path.join(finalDir,'presentation/project_presentation.pptx');
await finalizePresentation({workspaceDir:workspace,candidatePath:candidate,finalPath,
 pythonExecutable:runtime+'/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','2','--require-native-table-slide','5','--require-native-table-slide','9'],
 explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[2,5,9],requiredNativeChartOwnerSlides:[4,6,7,8,9],
 fontPolicy:{basis:'design',families:[FONT]},materializeLiteralChartWorkbooks:true,verifyArtifactToolImport:true,
 receiptPath:path.join(build,'validation_final.json')});
await fs.copyFile(new URL(import.meta.url),path.join(finalDir,'presentation/build_presentation.mjs'));
console.log('Final presentation',finalPath,'duration',notes.reduce((n,a)=>n+a.seconds,0));
