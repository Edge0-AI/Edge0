export type Stock = {sku:string;units:number};
export function buildStock(stock:Stock[]):Map<string,number>{
 const m=new Map<string,number>();
 for(const s of stock){
  const sku=s.sku.trim();
  if(!sku)throw new RangeError('empty sku');
  if(!Number.isInteger(s.units)||s.units<0)throw new RangeError('invalid units');
  m.set(sku,(m.get(sku)??0)+s.units);
 }
 return m;
}
