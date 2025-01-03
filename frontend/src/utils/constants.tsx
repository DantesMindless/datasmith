import DataSourcesTables from "../components/DataSourcesTables";
import CreateConnection from "../components/CreateConnection";


export const pageComponents = {
  createConnection: {
    name: "Create Connection",
    component: CreateConnection,
  },
  listConnections: {
    name: "List Connections",
    component: DataSourcesTables,
  },
};
