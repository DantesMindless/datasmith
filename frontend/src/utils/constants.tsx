import DataSourcesTables from "../components/DataSourcesTables";
import CreateConnection from "../components/CreateConnection";
import QueryTables from "../components/QueryTables";
import CreateModel from "../components/CreateModel"
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
    skip: false,
    name: "Query Tab",
    component: QueryTables,
  },
    createModel: {
    skip: true,
    name: "Create Model",
    component: CreateModel,
  },

};
