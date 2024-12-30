import { createContext } from 'react';
import type { ContextType } from './constants';

export const Context = createContext<ContextType | undefined>(undefined);
