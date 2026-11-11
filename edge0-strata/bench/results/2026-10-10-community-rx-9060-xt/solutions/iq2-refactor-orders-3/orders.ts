import { validateOrders } from './validation.ts';
import { aggregateOrders } from './aggregation.ts';

export type Order = { customer: string; priceCents: number; quantity: number; cancelled?: boolean };
export type Summary = { customer: string; totalCents: number; units: number };

export function summarizeOrders(orders: Order[]): Summary[] {
  return aggregateOrders(validateOrders(orders));
}
