import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const r='C:/Users/prani/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const {FileBlob,PresentationFile}=await import(pathToFileURL(r).href);
const p=await PresentationFile.importPptx(await FileBlob.load('D:/CV_Project/final_submission/presentation/project_presentation.pptx'));
for(let i=0;i<p.slides.items.length;i++){
 const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile('D:/CV_Project/results/submission_build/slides/final-'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await b.arrayBuffer()));
}
console.log('Rendered final slides',p.slides.items.length);
