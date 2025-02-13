// SQLFilterForm.tsx
import React, { useState, useEffect } from 'react';
import { Select, MenuItem, TextField, Box } from '@mui/material';
import { prepareFilterValue } from './Filters';
interface SQLFilterFormProps {
  fieldName: string;
  fieldType: string;
  onFilterChange: (whereClause: string) => void;
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
          '& .MuiSelect-select': { padding: '5px' }
        }}
      >
        {(operatorsByType[fieldType] || operatorsByType.string).map(op => (
          <MenuItem key={op} value={op}>
            {op}
          </MenuItem>
        ))}
      </Select>
      {renderValueInput()}
    </Box>
  );
};

export default SQLFilterForm;