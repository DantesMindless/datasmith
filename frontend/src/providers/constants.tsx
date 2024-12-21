import { ReactNode } from 'react';

interface AlertProviderProps {
    children: ReactNode;
}

// interface AlertData extends Record<string, any> {}

interface Alert {
    message: string;
    type: 'error' | 'success' | 'info' | 'warning';
}

interface AlertContextType {
    alert: Alert | null;
    showAlert: (message: string, type?: 'error' | 'success' | 'info' | 'warning') => void;
}

export type {AlertContextType, AlertProviderProps, Alert }
