/** Offline implementations of engine data access; original VoiceModel and TractMap are used. */
export {TractMap} from 'virtual:tract-map';
export const NEUTRAL:any={};
export const ARTIC_KEYS=['jaw','tbX','tbY','ttX','ttY','ttCurl','root','lipAp','lipProt','lipWidth','labiodental','velum','larynx','epilarynx','ventricular','lateral','dorsum'];
export const VOWELS:any[]=[]; export const VOWEL_TARGETS:any={};
let replay:any;
export function setReplay(value:any) {replay=value;}
export const airwayModel=() => replay;
export const dropAirwayModel=()=>{};
export const airwayFrame=(m:any)=>m.frameData;
export const airwayAreas=(m:any)=>m.areaData;
export const airwayTransfer=()=>({freqs:[],db:[],formants:[]});
