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
    filters: string[]
    initialLoad: boolean
    headers : string[]
}

type Connections = Connection[] | null

interface User {
    id: number;
    username: string;
    email: string;
}

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
    user: User | null;
    isAuthenticated: boolean;
    login: (username: string, password: string) => Promise<boolean>;
    logout: () => void;
}

export type {ContextType, ProviderProps, Alert, Info, Connections, Connection, TableViewTab, User }
