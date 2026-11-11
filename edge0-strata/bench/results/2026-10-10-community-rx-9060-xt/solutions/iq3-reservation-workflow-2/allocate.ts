import {buildStock} from './stock.ts';import type {Stock} from './stock.ts';
import {uniqueRequests} from './requests.ts';import type {Request} from './requests.ts';
export function reserve(stock:Stock[],requests:Request[]){
 const remaining=buildStock(stock);const accepted:string[]=[];const rejected:{id:string;reason:string}[]=[];
 for(const r of uniqueRequests(requests)){
  if(!remaining.has(r.sku)){rejected.push({id:r.id,reason:'unknown-sku'});continue;}
  const n=remaining.get(r.sku)!;
  if(n>=r.quantity){remaining.set(r.sku,n-r.quantity);accepted.push(r.id);}
  else rejected.push({id:r.id,reason:'insufficient-stock'});
 }
 const sorted=[...remaining].sort((a,b)=>a[0]<b[0]?-1:a[0]>b[0]?1:0);
 return {accepted,rejected,remaining:Object.fromEntries(sorted)};
}
