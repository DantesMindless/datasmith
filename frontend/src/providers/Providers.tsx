import React, { useState } from "react";
import type {
  Alert,
  Info,
  Connection,
  Connections,
  TableViewTab,
  ProviderProps,
} from "./constants";
import { getConnections } from "../utils/requests";
import { Context } from "./context";

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
    type: "error" | "success" | "warning" = "error"
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
        info,
        showInfo,
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
        setTabs
      }}
    >
      {children}
    </Context.Provider>
  );
};
