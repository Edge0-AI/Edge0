export type Stock = {sku:string;units:number};
export function trimSku(sku:string):string{
 if(typeof sku!=='string')throw new RangeError('sku must be a string');
 const t=sku.trim();
 if(t==='')throw new RangeError('sku must be nonempty');
 return t;
}
export function buildStock(stock:Stock[]):Map<string,number>{
 const m=new Map<string,number>();
 for(const s of stock){
  const sku=trimSku(s.sku);
  if(!Number.isInteger(s.units)||s.units<0)throw new RangeError('units must be a nonnegative integer');
  m.set(sku,(m.get(sku)??0)+s.units);
 }
 return m;
}
