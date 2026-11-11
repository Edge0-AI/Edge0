export type Stock = {sku:string;units:number};
export function buildStock(stock:Stock[]):Map<string,number>{
 const map=new Map<string,number>();
 for(const s of stock){
  if(typeof s.sku!=='string'||typeof s.units!=='number'||!Number.isInteger(s.units)||s.units<0)throw new RangeError('invalid stock record');
  const sku=s.sku.trim();
  if(sku==='')throw new RangeError('empty sku');
  map.set(sku,(map.get(sku)??0)+s.units);
 }
 return map;
}
