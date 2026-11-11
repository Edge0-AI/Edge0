import type {Order, Summary} from './orders.ts';

function compareCodePointOrder(a: string, b: string): number {
  const pa=[...a];
  const pb=[...b];
  for(let i=0;i<Math.max(pa.length,pb.length);i++){
    const x=pa[i];
    const y=pb[i];
    if(x===undefined)return -1;
    if(y===undefined)return 1;
    if(x!==y)return x<y?-1:1;
  }
  return 0;
}

export function aggregateOrders(orders: Order[]): Summary[] {
  const groups=new Map<string,Summary>();
  for(const o of orders){
    if(o.cancelled)continue;
    let s=groups.get(o.customer);
    if(!s){s={customer:o.customer,totalCents:0,units:0};groups.set(o.customer,s);}
    s.totalCents+=o.priceCents*o.quantity;
    s.units+=o.quantity;
  }
  return [...groups.values()].sort((a,b)=>compareCodePointOrder(a.customer,b.customer));
}
