import {buildStock} from './stock.ts';import type {Stock} from './stock.ts';
import {uniqueRequests} from './requests.ts';import type {Request} from './requests.ts';
export function reserve(stock:Stock[],requests:Request[]){
 const remaining=buildStock(stock);const accepted:string[]=[];const rejected:{id:string;reason:string}[]=[];
 for(const r of uniqueRequests(requests)){const n=remaining.get(r.sku)??0;remaining.set(r.sku,Math.max(0,n-r.quantity));if(n>=r.quantity)accepted.push(r.id);else rejected.push({id:r.id,reason:'insufficient-stock'});}
 return {accepted,rejected,remaining:Object.fromEntries(remaining)};
}
