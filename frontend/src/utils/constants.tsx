import DataSourcesTables from "../components/DataSourcesTables";
import CreateConnection from "../components/CreateConnection";
import QueryTables from "../components/QueryTables";

export const pageComponents = {
  createConnection: {
    skip: false,
    name: "Create Connection",
    component: CreateConnection,
  },
  listConnections: {
    skip: false,
    name: "List Connections",
    component: DataSourcesTables,
  },
  queryTab: {
    skip: true,
    name: "Query Tab",
    component: QueryTables,
  },
};
