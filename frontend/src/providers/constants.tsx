import React, { ReactNode } from 'react';

interface ProviderProps {
    children: ReactNode;
}

interface Alert {
    message: string;
    type: 'error' | 'success' | 'warning';
}

interface Info {
    message: string;
}

interface Connection {
    id: string;
    name: string;
    description: string;
    type: string;
    credentials?: Record<string, string | number>;
}

interface TableViewTab{
    ID: string  // datasource ID
    schema: string // database
    table: string // table name
    query: string
    
    joins: Record<string, string>[]
    page:   number
    perPage: number
    maxItems: number
    name: string
    data: Record<string, string | number | null>[]
    openedColumns: boolean
    columns: string[]
    activeColumns: string[]

    columnTypes: Record<string, string>
    column_filters: Record<string, string>
    where_clause: string

    initialLoad: boolean
    headers : string[]
    scrollState: ScrollState
}

interface ScrollState {
    newTab: boolean
    allDataLoaded: boolean
    scrollTop: number
}

interface FilterFields {
    value: string
    valueEnd: string
    operator: string
}

type Connections = Connection[] | null

interface ContextType {
    alert: Alert | null;
    showAlert: (message: string, type?: 'error' | 'success' | 'info' | 'warning') => void;
    info: Info | null;
    showInfo: (message: string) => void;
    connections: Connections | null;
    updateConnections:  ()=> void
    setActiveConnections:  React.Dispatch<React.SetStateAction<string[] | null>>
    activeConnections: string[] | null
    tabs: TableViewTab[] | null,
    setTabs: React.Dispatch<React.SetStateAction<TableViewTab[] | null>>
    addTableViewTab: (connection: Connection, schema: string, table: string) => void
    removeTab: (index: number) => void
}

interface Filter{
    clause: string;
    column: string;
}

export type {ContextType, ProviderProps, Alert, Info, 
    Connections, Connection, TableViewTab, Filter,FilterFields}//, Filter