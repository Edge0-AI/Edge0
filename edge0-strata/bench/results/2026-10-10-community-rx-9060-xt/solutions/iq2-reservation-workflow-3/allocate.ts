import {buildStock} from './stock.ts';import type {Stock} from './stock.ts';
import {uniqueRequests} from './requests.ts';import type {Request} from './requests.ts';
export function reserve(stock:Stock[],requests:Request[]){
 const remaining=buildStock(stock);
 const unique=uniqueRequests(requests);
 const accepted:string[]=[];const rejected:{id:string;reason:string}[]=[];
 for(const r of unique){
  if(!remaining.has(r.sku)){rejected.push({id:r.id,reason:'unknown-sku'});continue;}
  const n=remaining.get(r.sku)!;
  if(n>=r.quantity){remaining.set(r.sku,n-r.quantity);accepted.push(r.id);}
  else rejected.push({id:r.id,reason:'insufficient-stock'});
 }
 const out:Record<string,number>={};
 for(const k of [...remaining.keys()].sort())out[k]=remaining.get(k)!;
 return {accepted,rejected,remaining:out};
}
