// import { Filter, FilterOperator } from "../providers/constants";

// const getColumnType = (value: string | number | boolean | Date) => {
//   if (value === null || value === undefined) return "string";
//   if (typeof value === "number") return "number";
//   if (typeof value === "boolean") return "boolean";
//   const dateRegex = /^\d{4}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])$/;
//   if (typeof value === "string" && dateRegex.test(value)) {
//     return "date";
//   }
//   return "string";
// };

// export function prepareFilterValue(value:any): any {
//     const type = getColumnType(value);
//       switch (type) {
//         case 'date':
//           return `'${new Date(value).toISOString()}'`;
//         default:
//           return value;
//     };
//   }
  
//   export function combineFilters(filters: Filter[], conjunction: 'AND' | 'OR' = 'AND'): string {
//     console.log("filters =", filters);
//     if (!filters.length) return '';
    
//     const conditions = filters.map(filter => filter.value);
//     console.log("conditions =", conditions.join(` ${conjunction} `));
//     return conditions.join(` ${conjunction} `);
//   }

//   // export function combineFilters(filters: Filter[], conjunction: 'AND' | 'OR' = 'AND'): string {
//   //   console.log("filters =", filters);
//   //   if (!filters.length) return '';
    
//   //   const conditions = filters.map(filter => prepareFilterValue(filter));
//   //   console.log("conditions =", conditions.join(` ${conjunction} `));
//   //   return conditions.join(` ${conjunction} `);
//   // }