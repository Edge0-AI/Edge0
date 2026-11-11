import type {Order} from './orders.ts';

export function validateOrders(orders: Order[]): Order[] {
  if(!Array.isArray(orders))throw new RangeError('orders');
  return orders.map(o=>{
    if(!o || typeof o.customer!=='string' || !o.customer.trim() || !Number.isInteger(o.priceCents) || o.priceCents<0 || !Number.isInteger(o.quantity) || o.quantity<=0 || (o.cancelled!==undefined && typeof o.cancelled!=='boolean'))throw new RangeError('order');
    return {...o,customer:o.customer.trim()};
  });
}
