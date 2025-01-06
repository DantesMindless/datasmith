import React, { useState } from "react";
import type {
  Alert,
  Connection,
  Connections,
  TableViewTab,
  ProviderProps,
} from "./constants";
import { getConnections } from "../utils/requests";
import { Context } from "./context";

function getDataStorage(objName: string, defaultValue: any = []): [] {
  const storageObject: string | null = localStorage.getItem(objName);
  if (!storageObject) {
    return defaultValue;
  }
  return JSON.parse(storageObject);
}

function setDataStorage(objName: string, obj: any): any[] {
  const data = JSON.stringify(obj);
  localStorage.setItem(objName, data);
}

export const ContextProvider: React.FC<ProviderProps> = ({ children }) => {
  const [alert, setAlert] = useState<Alert | null>(null);
  const [connections, setConnections] = useState<Connections[]>(
    getDataStorage("connections")
  );
  const [activeConnections, setActiveConnections] = useState<string[]>(
    getDataStorage("activeConnections", {})
  );
  const [activePage, setActivePage] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<number | null>(null);
  const [tabs, setTabs] = useState<TableViewTab[]>(
    getDataStorage("activeTabs")
  );

  const showAlert = (
    message: string,
    type: "error" | "success" | "info" | "warning" = "error"
  ) => {
    setAlert({ message, type });
    setTimeout(() => setAlert(null), 5000);
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
      perPage: 100,
      maxItems: 0,
      name: `${schema} - ${table} #${tabsCount.length}`,
      data: []
    };
    tabs.push(tabObj);
    setDataStorage("activeTabs", [...tabs]);
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
      setActiveTab(tabs.length - 1)
      setActivePage("queryTab");
    }
  };

  const updateConnections = async () => {
    try {
      const data = await getConnections();
      console.log("connections", connections);
      setDataStorage("connections", [...data]);
      setConnections(data);
    } catch (error) {
      console.log(error);
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

  return (
    <Context.Provider
      value={{
        activeTab,
        alert,
        showAlert,
        connections,
        updateConnections,
        activeConnections,
        updateActiveConnections,
        updateActiveTab,
        tabs,
        addTableViewTab,
        removeTab,
        activePage,
        setActivePage,
      }}
    >
      {children}
    </Context.Provider>
  );
};
