import React, { ReactNode } from 'react';

interface ProviderProps {
    children: ReactNode;
}

interface Alert {
    message: string;
    type: 'error' | 'success' | 'info' | 'warning';
}

interface Connection {
    id: string;
    name: string;
    description: string;
    type: string;
    credentials?: Record<string, string | number>;
}

interface Tab{
    ID: string  // datasource ID
    schema: string // database
    table: string // table name
    query: string
    page:   number
    perPage: number
    maxItems: number
    name: string
    data: Record<string, string | number | null>[]
}

type Connections = Connection[] | null

interface ContextType {
    alert: Alert | null;
    showAlert: (message: string, type?: 'error' | 'success' | 'info' | 'warning') => void;
    connections: Connections | null;
    updateConnections:  ()=> void
    setActiveConnections:  React.Dispatch<React.SetStateAction<string[] | null>>
    activeConnections: string[] | null
    tabs: Tab[] | null,
    setTabs: React.Dispatch<React.SetStateAction<Tab[] | null>>
}

export type {ContextType, ProviderProps, Alert, ActiveConnection, Connections, Connection }
