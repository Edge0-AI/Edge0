export type Order = {customer: string; priceCents: number; quantity: number; cancelled?: boolean};
export type Summary = {customer: string; totalCents: number; units: number};
export function summarizeOrders(orders: Order[]): Summary[] {
  if(!Array.isArray(orders))throw new RangeError('orders');
  const normalized=orders.map(o=>{
    if(!o || typeof o.customer!=='string' || !o.customer.trim() || !Number.isInteger(o.priceCents) || o.priceCents<0 || !Number.isInteger(o.quantity) || o.quantity<=0 || (o.cancelled!==undefined && typeof o.cancelled!=='boolean'))throw new RangeError('order');
    return {...o,customer:o.customer.trim()};
  });
  const groups=new Map<string,Summary>();
  for(const o of normalized){if(o.cancelled)continue;let s=groups.get(o.customer);if(!s){s={customer:o.customer,totalCents:0,units:0};groups.set(o.customer,s);}s.totalCents+=o.priceCents*o.quantity;s.units+=o.quantity;}
  return [...groups.values()].sort((a,b)=>a.customer<b.customer?-1:a.customer>b.customer?1:0);
}
