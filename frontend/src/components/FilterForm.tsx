import React, { useState, useEffect, forwardRef, useImperativeHandle, useRef } from 'react';
import { Select, MenuItem, TextField, Box, Button } from '@mui/material';
import { FilterFields, Filter } from '../providers/constants';

interface SQLFilterFormProps {
  fieldName: string;
  fieldType: string;
  onFilterChange: (whereClause: string, filter: FilterFields) => void;
  initialValues: FilterFields;
}

export interface SQLFilterFormRef {
  resetForm: () => void;
  updateForm: (values: FilterFields) => void;
}

export function combineFilters(filters: Filter[], conjunction: 'AND' | 'OR' = 'AND'): string {
  if (!filters.length) return '';
  
  const conditions = filters.map(filter => filter.clause);
  return conditions.join(` ${conjunction} `);
}

const SQLFilterForm = forwardRef<SQLFilterFormRef, SQLFilterFormProps>(({ 
  fieldName, 
  fieldType = 'text',
  onFilterChange,
  initialValues
}, ref) => {
  const [filters, setFilters] = useState<FilterFields>({
    operator: initialValues?.operator || '=',
    value: initialValues?.value || '',
    valueEnd: initialValues?.valueEnd || ''
  });

  useImperativeHandle(ref, () => ({
    resetForm: () => {
      const defaultFilters = {
        operator: '=',
        value: '',
        valueEnd: ''
      };
      setFilters(defaultFilters);
      onFilterChange('', defaultFilters);
    }
  }));

  const operatorsByType = {
    text: ['=', '!=', 'LIKE', 'NOT LIKE', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'],
    number: ['=', '!=', '>', '<', '>=', '<=', 'BETWEEN', 'RANGE', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'],
    date: ['=', '!=', '>', '<', '>=', '<=', 'BETWEEN', 'IS NULL', 'IS NOT NULL'],
    boolean: ['=', 'IS NULL', 'IS NOT NULL'],
    time: ['=', '!=', '>', '<', '>=', '<=', 'BETWEEN', 'IS NULL', 'IS NOT NULL'],
    timestamp: ['=', '!=', '>', '<', '>=', '<=', 'BETWEEN', 'IS NULL', 'IS NOT NULL'],
    string: ['=', '!=', 'LIKE', 'NOT LIKE', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL'] // fallback for string type
  };

  const convertFilterFieldValue = (value:any, fieldType:string): any => {
    switch (fieldType) {
      case 'date':
        return `'${new Date(value).toISOString()}'`;
      default:
        return value;
    };
  }

  const generateWhereClause = () => {
    if (!filters.value && filters.operator !== 'IS NULL' && filters.operator !== 'IS NOT NULL' ) return '';
    if (isRangeOperator(filters.operator) && !filters.valueEnd) return '';
    
    let { operator, value, valueEnd } = filters;
    value = convertFilterFieldValue(value, fieldType);
    valueEnd = isRangeOperator(operator) ? convertFilterFieldValue(valueEnd, fieldType) : '';
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

  const lastWhereClause = useRef<string>('');

  useEffect(() => {
    if (filters.operator || filters.value || filters.valueEnd) {
      const whereClause = generateWhereClause();
      if (whereClause !== lastWhereClause.current) {
        lastWhereClause.current = whereClause;
        onFilterChange(whereClause, filters);
      }
    }
  }, [filters]);

  useEffect(() => {
    if (initialValues && (
      initialValues.operator !== filters.operator ||
      initialValues.value !== filters.value ||
      initialValues.valueEnd !== filters.valueEnd
    )) {
      setFilters({
        operator: initialValues.operator || '=',
        value: initialValues.value || '',
        valueEnd: initialValues.valueEnd || ''
      });
    }
  }, [initialValues]);

  useImperativeHandle(ref, () => ({
    resetForm: () => {
      const defaultFilters = {
        operator: '=',
        value: '',
        valueEnd: ''
      };
      setFilters(defaultFilters);
      lastWhereClause.current = '';
      onFilterChange('', defaultFilters);
    },
    updateForm: (newValues: FilterFields) => {
      setFilters(newValues);
      const whereClause = generateWhereClause();
      if (whereClause !== lastWhereClause.current) {
        lastWhereClause.current = whereClause;
        onFilterChange(whereClause, newValues);
      }
    }
  }));

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
});

export default SQLFilterForm;