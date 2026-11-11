export type Request = {id:string;sku:string;quantity:number};
export function uniqueRequests(requests:Request[]):Request[]{
 const out:Request[]=[];const seen=new Set<string>();
 for(const r of requests){
  if(typeof r.id!=='string'||typeof r.sku!=='string'||typeof r.quantity!=='number'||!Number.isInteger(r.quantity)||r.quantity<=0)throw new RangeError('invalid request record');
  const id=r.id.trim();const sku=r.sku.trim();
  if(id===''||sku==='')throw new RangeError('empty id or sku');
  if(seen.has(id))continue;
  seen.add(id);out.push({id,sku,quantity:r.quantity});
 }
 return out;
}
