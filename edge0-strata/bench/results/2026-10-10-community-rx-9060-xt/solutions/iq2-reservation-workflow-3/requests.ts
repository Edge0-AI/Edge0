export type Request = {id:string;sku:string;quantity:number};
export function uniqueRequests(requests:Request[]):Request[]{
 const seen=new Set<string>();const out:Request[]=[];
 for(const r of requests){
  const id=r.id.trim();const sku=r.sku.trim();
  if(!id)throw new RangeError('empty id');
  if(!sku)throw new RangeError('empty sku');
  if(!Number.isInteger(r.quantity)||r.quantity<=0)throw new RangeError('invalid quantity');
  if(seen.has(id))continue;
  seen.add(id);out.push({id,sku,quantity:r.quantity});
 }
 return out;
}
