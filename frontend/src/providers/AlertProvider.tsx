import React, { useState } from 'react';
import type { Alert } from './AlertContext';
import { AlertContext } from './AlertContext';
import type { AlertProviderProps } from './constants';


export const AlertProvider: React.FC<AlertProviderProps> = ({ children }) => {
    const [alert, setAlert] = useState<Alert | null>(null);

    const showAlert = (message: string, type: 'error' | 'success' | 'info' | 'warning' = 'error') => {
        setAlert({ message, type });
        // Auto-clear the alert after 5 seconds
        setTimeout(() => setAlert(null), 5000);
    };

    return (
        <AlertContext.Provider value={{ alert, showAlert }}>
            {children}
        </AlertContext.Provider>
    );
};
