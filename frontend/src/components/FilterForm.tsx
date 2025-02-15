// SQLFilterForm.tsx
import React, { useState, useEffect } from 'react';
import { Select, MenuItem, TextField, Box, Button } from '@mui/material';
//import { prepareFilterValue } from './Filters';

interface SQLFilterFormProps {
  fieldName: string;
  fieldType: string;
  onFilterChange: (whereClause: string) => void;
}


export function getColumnType(value: string | number | boolean | Date) :string {
  if (value === null || value === undefined) return "string";
  if (typeof value === "number") return "number";
  if (typeof value === "boolean") return "boolean";
  const dateRegex = /^\d{4}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])$/;
  if (typeof value === "string" && dateRegex.test(value)) {
    return "date";
  }
  return "string";
};

export function combineFilters(filters: Filter[], conjunction: 'AND' | 'OR' = 'AND'): string {
  console.log("filters =", filters);
  if (!filters.length) return '';
  
  const conditions = filters.map(filter => filter.clause);
  console.log("conditions =", conditions.join(` ${conjunction} `));
  return conditions.join(` ${conjunction} `);
}

const SQLFilterForm: React.FC<SQLFilterFormProps> = ({ 
  fieldName, 
  fieldType = 'text',
  onFilterChange 
}) => {
  const [filters, setFilters] = useState({
    operator: '=',
    value: '',
    valueEnd: '',
  });

  const operatorsByType = {
    text: ['=', '!=', 'LIKE', 'NOT LIKE', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'],
    number: ['=', '!=', '>', '<', '>=', '<=', 'BETWEEN', 'RANGE', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'],
    date: ['=', '!=', '>', '<', '>=', '<=', 'BETWEEN', 'IS NULL', 'IS NOT NULL'],
    boolean: ['=', 'IS NULL', 'IS NOT NULL'],
    time: ['=', '!=', '>', '<', '>=', '<=', 'BETWEEN', 'IS NULL', 'IS NOT NULL'],
    timestamp: ['=', '!=', '>', '<', '>=', '<=', 'BETWEEN', 'IS NULL', 'IS NOT NULL'],
    string: ['=', '!=', 'LIKE', 'NOT LIKE', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'] // fallback for string type
  };

  const prepareFilterValue = (value:any): any => {
    const type = getColumnType(value);
      switch (type) {
        case 'date':
          return `'${new Date(value).toISOString()}'`;
        default:
          return value;
    };
  }

  const generateWhereClause = () => {
    if (!filters.value) return '';
    
    let { operator, value, valueEnd } = filters;
    value = prepareFilterValue(value);
    valueEnd = prepareFilterValue(valueEnd);
    let whereClause = '';

    if (operator === 'IS NULL' || operator === 'IS NOT NULL') {
      whereClause = `${fieldName} ${operator}`;
    } else if (operator === 'LIKE' || operator === 'NOT LIKE') {
      whereClause = `${fieldName} ${operator} '%${value}%'`;
    } else if (operator === 'IN' || operator === 'NOT IN') {
      const values = value.split(',').map(v => v.trim());
      const formattedValues = fieldType === 'text' || fieldType === 'string' ? 
        values.map(v => `'${v}'`).join(', ') : 
        values.join(', ');
      whereClause = `${fieldName} ${operator} (${formattedValues})`;
    } else if (operator === 'BETWEEN') {
      const formatValue = (val) => fieldType === 'text' || fieldType === 'string' ? `'${val}'` : val;
      whereClause = `${fieldName} BETWEEN ${formatValue(value)} AND ${formatValue(valueEnd)}`;
    } else if (operator === 'RANGE') {
      whereClause = `${fieldName} >= ${value} AND ${fieldName} <= ${valueEnd}`;
    } else {
      const formattedValue = fieldType === 'text' || fieldType === 'string' ? `'${value}'` : value;
      whereClause = `${fieldName} ${operator} ${formattedValue}`;
    }

    return whereClause;
  };

  useEffect(() => {
      if (filters.operator || filters.value || filters.valueEnd) {
      const whereClause = generateWhereClause();
      onFilterChange(whereClause);
    }
  }, [filters]);

  const isRangeOperator = (op: string) => ['BETWEEN', 'RANGE'].includes(op);

  const renderValueInput = () => {
    if (filters.operator === 'IS NULL' || filters.operator === 'IS NOT NULL') {
      return null;
    }

    const commonProps = {
      size: 'small' as const,
      fullWidth: true,
      sx: { 
        '& .MuiInputBase-root': { height: '30px', fontSize: '12px' },
        '& .MuiInputBase-input': { padding: '5px' }
      }
    };

    const renderInput = (value: string, onChange: (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => void) => {
      switch (fieldType) {
        case 'boolean':
          return (
            <Select
              {...commonProps}
              value={value}
              onChange={onChange}
            >
              <MenuItem value="">All</MenuItem>
              <MenuItem value="true">True</MenuItem>
              <MenuItem value="false">False</MenuItem>
            </Select>
          );
        case 'date':
          return (
            <TextField
              {...commonProps}
              type="date"
              value={value}
              onChange={onChange}
            />
          );
        case 'time':
          return (
            <TextField
              {...commonProps}
              type="time"
              value={value}
              onChange={onChange}
            />
          );
        case 'timestamp':
          return (
            <TextField
              {...commonProps}
              type="datetime-local"
              value={value}
              onChange={onChange}
            />
          );
        case 'number':
          return (
            <TextField
              {...commonProps}
              type="number"
              value={value}
              onChange={onChange}
            />
          );
        default:
          return (
            <TextField
              {...commonProps}
              type="text"
              value={value}
              onChange={onChange}
              placeholder={filters.operator === 'IN' ? "Values, comma separated" : "Value..."}
            />
          );
      }
    };

    return (
      <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
        {renderInput(
          filters.value,
          (e) => setFilters({ ...filters, value: e.target.value })
          )}
        {isRangeOperator(filters.operator) && (
          renderInput(
            filters.valueEnd,
            (e) => setFilters({ ...filters, valueEnd: e.target.value })
          )
        )}
    </Box>
    );
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
      <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
        <Select
          size="small"
          value={filters.operator}
          onChange={(e) => setFilters({ 
            ...filters, 
            operator: e.target.value as string,
            valueEnd: '' 
          })}
          sx={{ 
            height: '30px',
            fontSize: '12px',
            flex: 1,
            '& .MuiSelect-select': { padding: '5px' }
          }}
        >
          {(operatorsByType[fieldType] || operatorsByType.string).map(op => (
            <MenuItem key={op} value={op}>
              {op}
            </MenuItem>
          ))}
        </Select>
        {(filters.value || filters.valueEnd) && (
        <Button 
          size="small"
          variant="contained"
          sx={{ 
            height: '30px',
            minWidth: '30px',
            fontSize: '12px',
            padding: '5px',
            marginLeft: 'auto'
          }}
          onClick={() => setFilters({ 
            operator: '=',
            value: '',
            valueEnd: '' 
          })}
        >X</Button>
      )}
      </Box>
      {renderValueInput()}
    </Box>
  );
};

export default SQLFilterForm;