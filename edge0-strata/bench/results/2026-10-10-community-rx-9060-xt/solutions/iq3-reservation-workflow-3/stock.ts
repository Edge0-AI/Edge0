export type Stock = {sku:string;units:number};
export function buildStock(stock:Stock[]):Map<string,number>{
 const map=new Map<string,number>();
 for(const s of stock){
  if(typeof s.sku!=='string'||s.sku.trim()==='')throw new RangeError('invalid stock sku');
  if(typeof s.units!=='number'||!Number.isInteger(s.units)||s.units<0)throw new RangeError('invalid stock units');
  const sku=s.sku.trim();map.set(sku,(map.get(sku)??0)+s.units);
 }
 return map;
}
