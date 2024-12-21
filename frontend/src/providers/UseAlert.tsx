import { useContext } from 'react';
import { AlertContext, AlertContextType } from './AlertContext';


export const useAlert = (): AlertContextType => {
    const context = useContext(AlertContext);
    if (context === undefined) {
        throw new Error('useAlert must be used within an AlertProvider');
    }
    return context;
};
