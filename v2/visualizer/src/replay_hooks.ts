/** Rendering environment only: upstream canvas callback bodies stay unchanged. */
export let current: any = null;
export function setCurrent(value:any) { current=value; }
export const engine = { get latest() { return current?.tele ?? null; } };
export const useStore = { getState: () => ({body:current?.model.body,artic:current?.artic}) };
export const useModel = () => ({ get current(){return current?.model;} });
let callback: any;
export function usePhosphor(draw:any) { callback=draw; return {current:null}; }
const refs=new Map<any,any[]>();let active:any,refIndex=0;
export function resetHooks(){refs.clear();}
export function useRef(value:any) {const list=refs.get(active)!;const i=refIndex++;return list[i]??(list[i]={current:value});}
export function runComponent(component:any,ctx:CanvasRenderingContext2D,w:number,h:number,dt:number) {
  active=component;refIndex=0;if(!refs.has(component))refs.set(component,[]);component(); callback({ctx,w,h,dt,dpr:1,now:0});
}
export function jsx(){return null;} export const jsxs=jsx; export const jsxDEV=jsx;
