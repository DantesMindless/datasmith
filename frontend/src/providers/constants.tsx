import { ReactNode } from 'react';

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

type Connections = Connection[] | null

interface ContextType {
    alert: Alert | null;
    showAlert: (message: string, type?: 'error' | 'success' | 'info' | 'warning') => void;
    connections: Connections | null;
    updateConnections: () => Promise<void>;
    activeConnection: string | null;
    setActiveConnection: () => Promise<void>;
}

export type {ContextType, ProviderProps, Alert, Connections, Connection }
