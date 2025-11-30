import httpfetch from './axios';
import { UUIDTypes } from 'uuid';
import { TableViewTab } from '../providers/constants';

async function getData(url: string){
  try {
    const response = await httpfetch.get(url);
    return response.data; // Returning fetched data
  } catch (error) {
    console.error("Error fetching data:", error);
    return [];
  }
}

async function putData(url: string, body: Record<string, string>){
  try {
    const response = await httpfetch.put(url, body);
    return response.data; // Returning fetched data
  } catch (error) {
    console.error("Error submitting data:", error);
    return [];
  }
}

export const getConnections = async () => {
  return getData("datasource/")
};

export const getSchemaTablesList = async (uuid : UUIDTypes, tableName:string) => {
  return getData(`datasource-schema-metadata/${uuid}/${tableName}/`)
};

export const getDatabasesList = async (uuid : UUIDTypes) => {
  return getData(`datasource-metadata/schemas/${uuid}/`)
};

export const getConnectionTypes = async () => {
  return getData("datasource/connection-types/")
};

export const queryTab = (tab: TableViewTab) => {
  // const getQuery = () => {
  //   return `SELECT * FROM ${tab.schema}.${tab.table} LIMIT ${tab.perPage};`;
  // }
  const query = tab
  return putData(`datasource/query/${tab.ID}/`, {query: query})
}

export const getJoins = (tab: TableViewTab) => {
  return getData(`datasource-metadata/tables/${tab.ID}/${tab.schema}/${tab.table}/`,)
}

export const exportTableToCSV = async (connectionId: string, exportParams: {
  schema: string;
  table: string;
  columns?: string[];
  filters?: string;
  limit?: number;
  dataset_name?: string;
  dataset_description?: string;
}) => {
  try {
    const response = await httpfetch.post(`datasource/export/${connectionId}/`, exportParams);
    return response.data;
  } catch (error) {
    console.error("Error exporting table:", error);
    throw error;
  }
}
