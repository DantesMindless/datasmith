import { createContext } from 'react';
import type { Alert, AlertContextType } from './constants';


const AlertContext = createContext<AlertContextType | undefined>(undefined);

export type {Alert, AlertContextType}
export { AlertContext }
