import DataSourcesTables from "../components/DataSourcesTables";
import Dashboard from "../components/Dashboard";


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
    component: DataSourcesTables,
  },
  listConnections: {
    name: "List Connections",
    component: DataSourcesTables,
  },
};
