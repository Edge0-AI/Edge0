import {buildStock} from './stock.ts';import type {Stock} from './stock.ts';
import {uniqueRequests} from './requests.ts';import type {Request} from './requests.ts';
function codePointCompare(a:string,b:string){const ca=Array.from(a);const cb=Array.from(b);const n=Math.min(ca.length,cb.length);for(let i=0;i<n;i++){if(ca[i]!==cb[i])return ca[i]<cb[i]?-1:1;}return ca.length-cb.length;}
export function reserve(stock:Stock[],requests:Request[]){
 const remaining=buildStock(stock);const accepted:string[]=[];const rejected:{id:string;reason:string}[]=[];
 for(const r of uniqueRequests(requests)){
  if(!remaining.has(r.sku)){rejected.push({id:r.id,reason:'unknown-sku'});continue;}
  const n=remaining.get(r.sku)!;
  if(n>=r.quantity){remaining.set(r.sku,n-r.quantity);accepted.push(r.id);}else rejected.push({id:r.id,reason:'insufficient-stock'});
 }
 const out:Record<string,number>={};
 for(const key of [...remaining.keys()].sort(codePointCompare))out[key]=remaining.get(key)!;
 return {accepted,rejected,remaining:out};
}
