export type Request = {id:string;sku:string;quantity:number};
export function uniqueRequests(requests:Request[]):Request[]{
 const out:Request[]=[];const seen=new Set<string>();
 for(const r of requests){
  if(typeof r?.id!=='string'||r.id.trim()===''||typeof r?.sku!=='string'||r.sku.trim()===''||!Number.isInteger(r.quantity)||r.quantity<=0)throw new RangeError('invalid request record');
  const id=r.id.trim();
  if(seen.has(id))continue;
  seen.add(id);
  out.push({id,sku:r.sku.trim(),quantity:r.quantity});
 }
 return out;
}
