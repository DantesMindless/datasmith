import DataSourcesTables from "../components/DataSourcesTables";
import Dashboard from "../components/Dashboard";
import CreateConnection from "../components/CreateConnection";

export const pageComponents = {
  home: {
    name: "Home",
    component: DataSourcesTables,
  },
  dashboard: {
    name: "Dashboard",
    component: Dashboard,
  },
  createConnection: {
    name: "Create Connection",
    component: CreateConnection,
  },
  listConnections: {
    name: "List Connections",
    component: DataSourcesTables,
  },
};

export const fieldTypes = {
  // CharField:
  // InteggerField:
};
