import {buildStock} from './stock.ts';import type {Stock} from './stock.ts';
import {uniqueRequests} from './requests.ts';import type {Request} from './requests.ts';
function codePointOrder(a:string,b:string){
 const ca=Array.from(a);const cb=Array.from(b);
 for(let i=0;i<Math.max(ca.length,cb.length);i++){
  const x=ca[i];const y=cb[i];
  if(x===undefined)return -1;if(y===undefined)return 1;
  if(x!==y)return x<y?-1:1;
 }
 return 0;
}
export function reserve(stock:Stock[],requests:Request[]){
 const remaining=buildStock(stock);const accepted:string[]=[];const rejected:{id:string;reason:string}[]=[];
 for(const r of uniqueRequests(requests)){
  const n=remaining.get(r.sku);
  if(n===undefined){rejected.push({id:r.id,reason:'unknown-sku'});continue;}
  if(n>=r.quantity){remaining.set(r.sku,n-r.quantity);accepted.push(r.id);}
  else rejected.push({id:r.id,reason:'insufficient-stock'});
 }
 const keys=[...remaining.keys()].sort(codePointOrder);
 return {accepted,rejected,remaining:Object.fromEntries(keys.map(k=>[k,remaining.get(k)!]))};
}
