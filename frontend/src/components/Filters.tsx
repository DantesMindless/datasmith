import { Filter, FilterOperator } from "../providers/constants";

function prepareFilterValue(filter: Filter): string | number | boolean | Date {
    const { value, from, to, operator, field, type } = filter;
  
    const formatValue = (val: any): any => {//string => {
      if (val === null || val === undefined) return 'NULL';
      
      switch (type) {
        case 'string':
          return `'${val.replace(/'/g, "''")}'`;
        case 'date':
          return `'${new Date(val).toISOString()}'`;
        case 'boolean':
          //return val ? '1' : '0';
          return val;// ? 'true' : 'false';
        case 'number':
          return val;//.toString();
        default:
          return `'${val}'`;
      }
    };
  
    switch (operator) {
      case FilterOperator.EQUALS:
        return `${field} = ${formatValue(value)}`;
      
      case FilterOperator.NOT_EQUALS:
        return `${field} != ${formatValue(value)}`;
      
      case FilterOperator.CONTAINS:
        if (type !== 'string') throw new Error('CONTAINS operator only supports string type');
        return `${field} LIKE '%${value.replace(/'/g, "''")}%'`;
      
      case FilterOperator.GREATER_THAN:
        return `${field} > ${formatValue(value)}`;
      
      case FilterOperator.LESS_THAN:
        return `${field} < ${formatValue(value)}`;
      
      case FilterOperator.BETWEEN:
        if (!from || !to) throw new Error('BETWEEN operator requires both from and to values');
        return `${field} BETWEEN ${formatValue(from)} AND ${formatValue(to)}`;
      
      case FilterOperator.IN:
        if (!Array.isArray(value)) throw new Error('IN operator requires array value');
        return `${field} IN (${value.map(formatValue).join(', ')})`;
      
      case FilterOperator.IS_NULL:
        return `${field} IS NULL`;
      
      case FilterOperator.IS_NOT_NULL:
        return `${field} IS NOT NULL`;
      
      default:
        throw new Error(`Unsupported operator: ${operator}`);
    }
  }
  
  export function combineFilters(filters: Filter[], conjunction: 'AND' | 'OR' = 'AND'): string {
    console.log("filters =", filters);
    if (!filters.length) return '';
    
    const conditions = filters.map(filter => prepareFilterValue(filter));
    console.log("conditions =", conditions.join(` ${conjunction} `));
    return conditions.join(` ${conjunction} `);
  }