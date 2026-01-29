import React, { useState, useEffect } from "react";
import type {
  Alert,
  Info,
  Connection,
  Connections,
  TableViewTab,
  ProviderProps,
  User,
} from "./constants";
import { getConnections } from "../utils/requests";
import { Context } from "./context";
import httpfetch, { setTokens, clearTokens, getAccessToken } from "../utils/axios";

function getDataStorage(objName: string, defaultValue: any = []): [] {
  const storageObject: string | null = sessionStorage.getItem(objName);
  if (!storageObject) {
    return defaultValue;
  }
  return JSON.parse(storageObject);
}

function setDataStorage(objName: string, obj: any): any[] {
  const data = JSON.stringify(obj);
  sessionStorage.setItem(objName, data);
}

export const ContextProvider: React.FC<ProviderProps> = ({ children }) => {
  const [alert, setAlert] = useState<Alert | null>(null);
  const [info, setInfo] = useState<Info | null>(null);
  const [connections, setConnections] = useState<Connections[] | null>(
    null
  );
  const [activeConnections, setActiveConnections] = useState<string[]>(
    getDataStorage("activeConnections", {})
  );
  const [activePage, setActivePage] = useState<string | null>("listConnections");
  const [activeTab, setActiveTab] = useState<number | null>(null);
  const [tabs, setTabs] = useState<TableViewTab[]>(
    getDataStorage("activeTabs")
  );
  const [user, setUser] = useState<User | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [authLoading, setAuthLoading] = useState<boolean>(true);
  const [cudaEnabled, setCudaEnabled] = useState<boolean>(() => {
    const stored = localStorage.getItem('cudaEnabled');
    return stored === 'true';
  });

  // Persist CUDA preference
  useEffect(() => {
    localStorage.setItem('cudaEnabled', String(cudaEnabled));
  }, [cudaEnabled]);

  useEffect(() => {
    const accessToken = getAccessToken();
    if (accessToken) {
      // Validate token before setting authenticated state
      httpfetch.get('test_token/')
        .then((response) => {
          setUser(response.data.user || null);
          setIsAuthenticated(true);
          // Load connections after successful authentication
          updateConnections();
        })
        .catch(() => {
          // Token is invalid, clear it
          clearTokens();
          setUser(null);
          setIsAuthenticated(false);
        })
        .finally(() => {
          setAuthLoading(false);
        });
    } else {
      setAuthLoading(false);
    }
  }, []);

  const showAlert = (
    message: string,
    type: "error" | "success" | "warning" | "info" = "error"
  ) => {
    setAlert({ message, type });
    setTimeout(() => setAlert(null), 5000);
  };

  const showInfo = (message: string) => {
      setInfo({ message });
      setTimeout(() => setInfo(null), 5000);
  };

  const addTableViewTab = async (
    connection: Connection,
    schema: string,
    table: string
  ) => {
    const tabsCount: TableViewTab[] = tabs.filter(
      (row) => schema === row.schema && table === row.table
    );
    const tabObj: TableViewTab = {
      ID: connection.id,
      schema: schema,
      table: table,
      query: `SELECT * FROM ${schema}.${table} LIMIT 100;`,
      page: 1,
      joins: [],
      perPage: 100,
      maxItems: 0,
      name: `${schema} - ${table} #${tabsCount.length}`,
      data: [],
      openedColumns: false,
      columns : [],
      activeColumns:[],
      filters: [],
      initialLoad: true,
      headers: []
    };
    tabs.push(tabObj);
    const tabsStorage = [...tabs]
    tabsStorage.forEach((row)=>{
      row.data.joins = {}
      row.data.data = []
    })
    setDataStorage("activeTabs", [...tabsStorage]);
    setTabs([...tabs]);
    setActiveTab(tabs.length -1)
  };

  const removeTab = async (index: number) => {
    const activeTabIndex = index - 1;
    updateActiveTab(activeTabIndex >= 0 ? activeTabIndex : null);
    if (tabs != null) {
      tabs.splice(index, 1);
      setDataStorage("activeTabs", [...tabs]);
      setTabs([...tabs]);
      if (tabs.length > 0){
        setActiveTab(tabs.length - 1)
        setActivePage("queryTab");
      }
      else{
        setActivePage("listConnections")
      }
    }
  };

  const updateConnections = async () => {
    try {
      const data = await getConnections();
      setDataStorage("connections", [...data]);
      setConnections(data);
    } catch (error) {
      showAlert("Error updating connections", "error");
    }
  };

  const updateActiveConnections = (activeConnectionsObj: string[]) => {
    const data = { ...activeConnectionsObj };
    setDataStorage("activeConnections", data);
    setActiveConnections(data);
  };

  const updateActiveTab = (index: number | null) => {
    setActiveTab(index);
  };

  const login = async (username: string, password: string): Promise<boolean> => {
    try {
      const response = await httpfetch.post('login/', {
        username,
        password
      });
      
      if (response.data.access && response.data.refresh) {
        setTokens(response.data.access, response.data.refresh);
        setUser(response.data.user);
        setIsAuthenticated(true);
        showAlert('Login successful!', 'success');
        // Load connections after successful login
        updateConnections();
        return true;
      }
      return false;
    } catch (error: any) {
      showAlert(error.response?.data?.error || 'Login failed', 'error');
      return false;
    }
  };

  const logout = async () => {
    try {
      const refreshToken = localStorage.getItem('refresh_token');
      if (refreshToken) {
        await httpfetch.post('logout/', { refresh: refreshToken });
      }
    } catch (error) {
      // Continue with logout even if request fails
    }
    
    clearTokens();
    setUser(null);
    setIsAuthenticated(false);
    showAlert('Logged out successfully', 'success');
  };

  return (
    <Context.Provider
      value={{
        alert,
        showAlert,
        info,
        showInfo,
        connections,
        updateConnections,
        activeConnections,
        setActiveConnections,
        updateActiveConnections,
        tabs,
        setTabs,
        addTableViewTab,
        removeTab,
        activePage,
        setActivePage,
        activeTab,
        updateActiveTab,
        user,
        isAuthenticated,
        authLoading,
        login,
        logout,
        cudaEnabled,
        setCudaEnabled
      }}
    >
      {children}
    </Context.Provider>
  );
};
