export type Request = {id:string;sku:string;quantity:number};
export function uniqueRequests(requests:Request[]):Request[]{
 const out:Request[]=[];const seen=new Set<string>();
 for(const r of requests){
  if(typeof r.id!=='string'||r.id.trim()==='')throw new RangeError('invalid request id');
  if(typeof r.sku!=='string'||r.sku.trim()==='')throw new RangeError('invalid request sku');
  if(typeof r.quantity!=='number'||!Number.isInteger(r.quantity)||r.quantity<=0)throw new RangeError('invalid request quantity');
  const id=r.id.trim();if(seen.has(id))continue;seen.add(id);
  out.push({id,sku:r.sku.trim(),quantity:r.quantity});
 }
 return out;
}
