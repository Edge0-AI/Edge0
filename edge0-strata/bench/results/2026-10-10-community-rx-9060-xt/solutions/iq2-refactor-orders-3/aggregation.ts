import type { Order, Summary } from './orders.ts';

export function aggregateOrders(orders: Order[]): Summary[] {
  const groups = new Map<string, Summary>();
  for (const o of orders) {
    if (o.cancelled) continue;
    let s = groups.get(o.customer);
    if (!s) {
      s = { customer: o.customer, totalCents: 0, units: 0 };
      groups.set(o.customer, s);
    }
    s.totalCents += o.priceCents * o.quantity;
    s.units += o.quantity;
  }
  return [...groups.values()].sort((a, b) => (a.customer < b.customer ? -1 : a.customer > b.customer ? 1 : 0));
}
