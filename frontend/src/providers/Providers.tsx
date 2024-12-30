import React, { useState } from 'react';
import type { Alert, Connections } from './constants';
import type { ProviderProps } from './constants';
import { getConnections } from '../utils/requests';
import { Context } from './context'

export const ContextProvider: React.FC<ProviderProps> = ({ children }) => {
    const [alert, setAlert] = useState<Alert | null>(null);
    const [connections, setConnections] = useState<Connections | null>(null);
    const [activeConnection, setActiveConnection] = useState<React.FC | null>(null)

    const showAlert = (message: string, type: 'error' | 'success' | 'info' | 'warning' = 'error') => {
        setAlert({ message, type });
        setTimeout(() => setAlert(null), 5000);
    };

    const updateConnections = async () => {
        try {
            const data = await getConnections();
            setConnections(data);
        } catch (error) {
            console.log(error)
            showAlert('Error updating connections', 'error');
        }
    };

    return (
        <Context.Provider value={{ alert, showAlert, connections, updateConnections, activeConnection, setActiveConnection}}>
            {children}
        </Context.Provider>
    );
};
