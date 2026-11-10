import "../vendor/chimera/src/ui/styles.css";
import {AnatomyRenderer, type ViewFlags} from "../vendor/chimera/src/ui/render/anatomy.ts";
import {Glow,label} from "../vendor/chimera/src/ui/render/vector.ts";
import {V} from "../vendor/chimera/src/ui/colors.ts";
import {VoiceModel} from "../vendor/chimera/src/ui/analysis.ts";
import {TubeView} from "../vendor/chimera/src/ui/components/TubeView.tsx";
import {Stroboscope} from "../vendor/chimera/src/ui/components/Stroboscope.tsx";
import {TractMap,setReplay} from './replay_engine';
import {setCurrent,runComponent,resetHooks} from './replay_hooks';
import type {HumanArtic,HumanFrame,Vec} from "../vendor/chimera/src/wasm/types.ts";
const WIDTH=1920,HEIGHT=1080,FPS=30,TILE_W=960,TILE_H=680;
const NAMES=['jaw','tbX','tbY','ttX','ttY','ttCurl','root','lipAp','lipProt','lipWidth','labiodental','velum','larynx','epilarynx','ventricular','lateral','dorsum'] as const;
const VIEW:ViewFlags={pressure:false,particles:false,grid:true,calipers:true};
type Point=[number,number];
type Part={id:string;name:string;body_scale:number;body:any;scale:number;n_ph:number;states:number[][];outer:Point[][];inner:Point[][];areas:number[][];lengths:number[][];ph_areas:number[][];or_areas:number[][];nasal_areas:number[][];nasal_lengths:number[][];velum:number[];levels:number[];f0:number[];glottal_area:number[][];glottal_flow:number[][]};
type Input={version:number;state_rate:number;levels_rate:number;bpm:number;beats_per_bar:number;total_beats:number;song_seconds:number;audio_seconds:number;parts:Part[]};
type Runtime={part:Part;canvas:HTMLCanvasElement;ctx:CanvasRenderingContext2D;renderer:AnatomyRenderer;model:VoiceModel};
const stage=document.querySelector<HTMLCanvasElement>('#stage')!,stageCtx=stage.getContext('2d',{alpha:false})!;
let input:Input,stateHz=30,lastTime=0,runtime:Runtime[]=[];
const glow=new Glow();
function idx(series:any[],t:number){return Math.min(series.length-1,Math.max(0,Math.round(t*stateHz)));}
function rowAt(series:number[][],t:number){return series[idx(series,t)];}
function stateAt(p:Part,t:number){return rowAt(p.states,t);}
function pointSeriesAt(series:Point[][],t:number):Vec[]{return series[idx(series,t)].map(p=>({x:p[0],y:p[1]}));}
function scalarAt(series:number[],t:number){return series[idx(series,t)];}
function mapPoint(x:number,y:number,scale:number):Vec{return {x:x*scale,y:y*scale};}
function buildArtic(states:number[]):HumanArtic{const a={} as HumanArtic;NAMES.forEach((n,i)=>a[n]=states[i]);return a;}
function buildFrame(part: Part, t: number, a: HumanArtic): HumanFrame {
  const scale = part.body.scale;
  const outer = pointSeriesAt(part.outer, t);
  const inner = pointSeriesAt(part.inner, t);
  const areas = rowAt(part.areas,t);
  const lengths = rowAt(part.lengths,t);
  const splitIndex = Math.floor(inner.length / 2);
  const center = mapPoint(a.tbX, a.tbY - 0.25 * a.jaw, scale);
  const tongueRadius = 2.25 * scale;
  const tip = mapPoint(a.ttX, a.ttY, scale);
  const tongueTop = inner.slice(Math.max(0, splitIndex - 1));
  const tongueBottom: Vec[] = [];
  for (let i = 0; i <= 24; i++) {
    const angle = -Math.PI * i / 24;
    tongueBottom.push({ x: center.x + Math.cos(angle) * tongueRadius, y: center.y + Math.sin(angle) * tongueRadius * 0.55 });
  }
  const tongue = [...tongueTop, ...tongueBottom];

  const upperLip: Vec[] = [];
  const lowerLip: Vec[] = [];
  const lipX = 8.35 * scale;
  const mouth = Math.max(0.04, Math.min(2.0, a.lipAp + 0.35 * a.lipProt));
  for (let i = 0; i <= 14; i++) {
    const u = i / 14;
    const x = (6.45 + 2.75 * u) * scale;
    const y = (8.95 + 0.10 * Math.sin(u * Math.PI) - 0.10 * a.labiodental * u) * scale;
    upperLip.push({ x, y: y + (1 - u) * mouth * 0.22 * scale });
    lowerLip.push({ x, y: y - (1 - u) * mouth * 0.35 * scale - 0.05 * scale });
  }
  // The outer half of the lip vectors continues into the source renderer's face profile.
  for (let i = 0; i < 7; i++) {
    const u = i / 6;
    upperLip[8 + i] = mapPoint(8.5 - 1.15 * u, 9.0 + 2.0 * u, scale);
    lowerLip[8 + i] = mapPoint(8.45 - 1.0 * u, 8.85 - 1.5 * u, scale);
  }
  const upperTeeth = [mapPoint(6.15, 9.50, scale), mapPoint(6.35, 9.75, scale), mapPoint(7.0, 9.73, scale), mapPoint(7.2, 9.48, scale)];
  const lowerTeeth = [mapPoint(6.2, 8.55 - a.jaw * 0.4, scale), mapPoint(6.4, 8.78 - a.jaw * 0.4, scale), mapPoint(7.0, 8.78 - a.jaw * 0.4, scale), mapPoint(7.2, 8.50 - a.jaw * 0.4, scale)];
  const jawBone = [mapPoint(6.9, 8.7 - a.jaw * 0.65, scale), mapPoint(5.6, 6.0 - a.jaw * 0.35, scale), mapPoint(4.3, 5.2, scale), mapPoint(4.0, 4.8, scale)];
  const g = mapPoint(-0.8, 0.2 + a.larynx * 0.25, scale);
  const epiglottis = [mapPoint(0.0, 2.7, scale), mapPoint(0.45, 1.8, scale), mapPoint(0.15, 0.8, scale), mapPoint(-0.2, 0.3, scale)];
  const velumY = 10.0 - scalarAt(part.velum, t) * 0.9;
  const velum = [1.0, 1.8, 2.7, 3.6, 4.4, 5.1].map((x, i) => mapPoint(x, velumY + 0.5 * Math.sin(i / 5 * Math.PI), scale));
  const grid = Array.from({ length: areas.length }, (_, i) => {
    const index = Math.round((i + 0.5) * (inner.length - 1) / areas.length);
    const inside = inner[index], outside = outer[index];
    const sectionLength = lengths[i];
    return { inner: inside, outer: outside, mid: { x: (inside.x + outside.x) / 2, y: (inside.y + outside.y) / 2 },
      d: Math.hypot(outside.x - inside.x, outside.y - inside.y), area: areas[i], s: lengths.slice(0,i).reduce((s,x)=>s+x,0)+sectionLength/2,
      region: (i < part.n_ph ? 0 : 1) as 0 | 1 | 2 | 3, ray: i };
  });
  const pharynxLength = lengths.slice(0, part.n_ph).reduce((s, x) => s + x, 0);
  const lipS = lengths.reduce((s, x) => s + x, 0);
  return {
    outer, inner, tongue, velum, upperLip, lowerLip, upperTeeth, lowerTeeth, jawBone, epiglottis,
    nasalCavity: [], glottis: g, grid, length: lipS, pharynxLength, teethS: lipS * 0.82, lipS,
    velumPort: scalarAt(part.velum, t), lipArea: Math.PI / 4 * mouth * Math.max(0.2, a.lipWidth) * scale * scale,
    tongueCenter: center, tongueRadius, tip, splitIndex,
  };
}


/** Reconstitute the actual upstream VoiceModel shape and original TractMap, every frame. */
function modelAt(p:Part,t:number,canonical=false):{model:VoiceModel;frame:HumanFrame;artic:HumanArtic;tele:any}{
  const artic=buildArtic(stateAt(p,t)),frame=buildFrame(p,t,artic);
  const lengths=rowAt(p.lengths,t),phLen=lengths.slice(0,p.n_ph).reduce((s,v)=>s+v,0),orLen=lengths.slice(p.n_ph).reduce((s,v)=>s+v,0);
  const areaData={phAreas:rowAt(p.ph_areas,t),orAreas:rowAt(p.or_areas,t),phLen,orLen};
  const replay={body:p.body,nPh:p.n_ph,nOr:p.areas[0].length-p.n_ph,pharynxBranches:{base:0,n:p.n_ph},oralBranches:{base:p.n_ph,n:p.areas[0].length-p.n_ph},glottisPort:-1,frameData:frame,areaData};
  let model:VoiceModel;
  if(canonical){setReplay(replay);model=new VoiceModel({female:0,child:0},96000);}
  else {
    model=Object.assign(Object.create(VoiceModel.prototype),{body:{female:0,child:0},bodyParams:p.body,tract:new TractMap(p.body),vowels:[],nPh:p.n_ph,nOr:p.areas[0].length-p.n_ph,frame:()=>frame,areas:()=>areaData});
  }
  const tele={body:model.body,t:{artic:Float32Array.from(stateAt(p,t)),...areaData,velumPort:scalarAt(p.velum,t),glottalArea:rowAt(p.glottal_area,t),glottalFlow:rowAt(p.glottal_flow,t),f0:scalarAt(p.f0,t),pressureRms:new Float32Array(p.areas[0].length),flow:new Float32Array(p.areas[0].length),noiseLevel:0,noiseNode:-1,gate:1,voicing:scalarAt(p.f0,t)>0?1:0}};
  return {model,frame,artic,tele};
}
function makeRuntime(p:Part):Runtime{
  const canvas=document.createElement('canvas');canvas.width=TILE_W;canvas.height=TILE_H;
  const renderer=new AnatomyRenderer();renderer.resize(TILE_W,TILE_H);
  return {part:p,canvas,ctx:canvas.getContext('2d')!,renderer,model:modelAt(p,0).model};
}
function paint(p:Part,t:number,ctx:CanvasRenderingContext2D,r:AnatomyRenderer,canonical=false){
  const value=modelAt(p,t,canonical);ctx.setTransform(1,0,0,1,0,0);ctx.fillStyle='#000';ctx.fillRect(0,0,TILE_W,TILE_H);
  // Original renderer reserves a readout column; translation centres the head in this tile.
  const middle={x:1.375*p.body.scale,y:8*p.body.scale};
  const fit=.88;ctx.setTransform(fit,0,0,fit,TILE_W/2-r.X(middle)*fit,TILE_H/2-r.Y(middle)*fit);
  r.draw(ctx,value.model,value.frame,value.artic,null,VIEW,{hover:null,drag:null,dt:0,formants:[],slots:[],info:[]});
}
function panel(x:number,y:number,w:number,h:number){stageCtx.strokeStyle=V.frame;stageCtx.lineWidth=1;stageCtx.strokeRect(x,y,w,h);}
function drawAt(t:number){
  t=Math.max(0,Math.min(input.audio_seconds,t));const dt=1/FPS;lastTime=t;
  stageCtx.setTransform(1,0,0,1,0,0);stageCtx.globalAlpha=1;stageCtx.fillStyle='#000';stageCtx.fillRect(0,0,WIDTH,HEIGHT);
  const lead=runtime.find(p=>p.part.id==='lead')??runtime[0],others=runtime.filter(x=>x!==lead);
  const slots=[{entity:lead,x:12,y:12,w:1080,h:704}];
  const cols=others.length>3?2:1,rows=Math.max(1,Math.ceil(others.length/cols)),cw=800/cols,ch=966/rows;
  others.forEach((entity,i)=>slots.push({entity,x:1104+(i%cols)*cw,y:12+Math.floor(i/cols)*ch,w:cw-8,h:ch-10}));
  for(const {entity:e,x,y,w,h} of slots){
    panel(x,y,w,h);paint(e.part,t,e.ctx,e.renderer);stageCtx.globalAlpha=Math.max(.3,Math.min(1,scalarAt(e.part.levels,t)));
    const dw=Math.min(w-4,(h-35)*TILE_W/TILE_H),dh=dw*TILE_H/TILE_W;stageCtx.drawImage(e.canvas,x+(w-dw)/2,y+(h-35-dh)/2,dw,dh);label(stageCtx,e.part.name,x+w/2,y+h-10,{size:16,color:V.blue,align:'center',ls:.1});stageCtx.globalAlpha=1;
  }
  const leadValue=modelAt(lead.part,t);setCurrent(leadValue);
  for(const [component,x,w] of [[TubeView,12,615],[Stroboscope,639,453]] as const){
    const c=document.createElement('canvas');c.width=w;c.height=258;const ctx=c.getContext('2d')!;
    runComponent(component,ctx,w,258,dt);stageCtx.globalAlpha=Math.max(.3,scalarAt(lead.part.levels,t));stageCtx.drawImage(c,x,732);stageCtx.globalAlpha=1;panel(x,732,w,258);
  }
  const beat=t*input.bpm/60,bar=Math.min(Math.ceil(input.total_beats/input.beats_per_bar),Math.floor(beat/input.beats_per_bar)+1);
  stageCtx.fillStyle=V.blueDim;stageCtx.fillRect(60,1015,1800,2);stageCtx.fillStyle=V.blue;stageCtx.fillRect(60,1014,1800*Math.min(1,beat/input.total_beats),4);
  label(stageCtx,`BAR ${String(bar).padStart(2,'0')}`,WIDTH/2,1051,{size:17,color:V.blue,align:'center',ls:.12});
  glow.apply(stage,stageCtx,1);
  stageCtx.save();stageCtx.setTransform(1,0,0,1,0,0);stageCtx.globalAlpha=1;
  stageCtx.font='14px sans-serif';stageCtx.textAlign='right';stageCtx.fillStyle='#c4c4c4';
  stageCtx.fillText('Visualization: CHIMERA vocal synth © 2026 Jameson Hodge (MIT)',WIDTH-16,HEIGHT-10);
  stageCtx.restore();
}
async function loadInput(data:Input){if(!data.parts.length)throw Error('No throat parts');input=data;stateHz=data.state_rate;reset();await document.fonts.ready;drawAt(0);}
function reset(){resetHooks();runtime=input.parts.map(makeRuntime);lastTime=0;}
function capture(c:HTMLCanvasElement=stage){return c.toDataURL('image/png');}
function compare(a:ImageData,b:ImageData){let changed=0,sum=0,max=0,channels=0;for(let i=0;i<a.data.length;i+=4){let pixel=false;for(let k=0;k<4;k++){const d=Math.abs(a.data[i+k]-b.data[i+k]);if(d){pixel=true;channels++;}sum+=d;max=Math.max(d,max);}if(pixel)changed++;}return {width:a.width,height:a.height,pixels:a.width*a.height,changed_pixel_count:changed,changed_channel_values:channels,exact_pixel_match_percent:100*(1-changed/(a.width*a.height)),mean_absolute_channel_difference:sum/a.data.length,maximum_channel_difference:max};}
function pixelComparison(){
  const part=input.parts.find(p=>p.id==='lead')??input.parts[0];const t=Math.min(10,input.song_seconds/2);
  const canvases=[false,true].map(canonical=>{const c=document.createElement('canvas');c.width=TILE_W;c.height=TILE_H;const r=new AnatomyRenderer();r.resize(TILE_W,TILE_H);paint(part,t,c.getContext('2d')!,r,canonical);new Glow().apply(c,c.getContext('2d')!,1);return c;});
  const a=canvases[0].getContext('2d')!.getImageData(0,0,TILE_W,TILE_H),b=canvases[1].getContext('2d')!.getImageData(0,0,TILE_W,TILE_H);
  const vis=document.createElement('canvas');vis.width=TILE_W*3;vis.height=TILE_H;const ctx=vis.getContext('2d')!;ctx.drawImage(canvases[1],0,0);ctx.drawImage(canvases[0],TILE_W,0);
  const diff=ctx.createImageData(TILE_W,TILE_H);for(let i=0;i<a.data.length;i+=4){diff.data[i]=Math.max(...[0,1,2].map(k=>Math.abs(a.data[i+k]-b.data[i+k])))*8;diff.data[i+3]=255;}ctx.putImageData(diff,2*TILE_W,0);
  return {source:capture(canvases[1]),single:capture(canvases[0]),visual:capture(vis),stats:compare(a,b),part:{id:part.id,name:part.name,throat:1},seconds:t};
}
function validate(){
  const results:any={};
  for(const part of input.parts){
    let max=0,changed=0,deltas:number[]=[],pixelDeltas:number[]=[],previous:ImageData|undefined;
    for(let i=0;i<part.areas.length;i++){
      const t=i/stateHz,value=modelAt(part,t);const areas=[...value.model.areas(value.artic).phAreas,...value.model.areas(value.artic).orAreas];
      for(let j=0;j<areas.length;j++)max=Math.max(max,Math.abs(areas[j]-part.areas[i][j]),Math.abs(value.frame.grid[j].area-part.areas[i][j]));
      if(i){const delta=part.areas[i].reduce((s,x,j)=>s+Math.abs(x-part.areas[i-1][j]),0)/areas.length;deltas.push(delta);if(delta>1e-8)changed++;}
      // Rasterise the real saved airway alone: grid rays inside/outer, constant brightness.
      const canvas=document.createElement('canvas');canvas.width=320;canvas.height=280;const ctx=canvas.getContext('2d')!;ctx.fillStyle='#000';ctx.fillRect(0,0,320,280);
      ctx.strokeStyle='#fff';ctx.lineWidth=1;ctx.beginPath();for(const g of value.frame.grid){ctx.moveTo(55+g.inner.x*24,260-g.inner.y*24);ctx.lineTo(55+g.outer.x*24,260-g.outer.y*24);}ctx.stroke();
      const data=ctx.getImageData(0,0,320,280);if(previous){let sum=0;for(let k=0;k<data.data.length;k+=4)sum+=Math.abs(data.data[k]-previous.data[k]);pixelDeltas.push(sum/(320*280));}previous=data;
    }
    results[part.id]={maximum_area_error_cm2:max,nonempty_grid:true,frames:part.areas.length,changed_area_transitions:changed,mean_absolute_area_change_cm2:deltas.reduce((a,b)=>a+b,0)/Math.max(1,deltas.length),maximum_absolute_area_change_cm2:Math.max(...deltas),airway_raster_mean_absolute_pixel_change:pixelDeltas.reduce((a,b)=>a+b,0)/Math.max(1,pixelDeltas.length),airway_raster_changed_transitions:pixelDeltas.filter(x=>x>0).length,brightness_min:Math.min(...part.levels),brightness_max:Math.max(...part.levels)};
  }
  return {method:'All replay frames: original model areas and measured grid versus solver-derived input; independent constant-brightness airway raster per frame.',parts:results};
}
(window as any).chimeraMulti={loadInput,drawAt,capture:()=>capture(),pixelComparison,validate,reset,dimensions:()=>({width:WIDTH,height:HEIGHT,fps:FPS,audio_seconds:input?.audio_seconds??0})};
(window as any).chimeraReady=true;
