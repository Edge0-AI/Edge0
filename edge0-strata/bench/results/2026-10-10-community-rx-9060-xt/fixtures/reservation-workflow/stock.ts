export type Stock = {sku:string;units:number};
export function buildStock(stock:Stock[]):Map<string,number>{return new Map(stock.map(s=>[s.sku,s.units]));}
