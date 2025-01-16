import httpfetch from './axios';
import { UUIDTypes } from 'uuid';
import { TableViewTab } from '../providers/constants';

const uname = "u@u.com";
const pass = "password";

async function getData(url: string){
  try {
    const response = await httpfetch.get(url, {
      auth: {
        username: uname,
        password: pass,
      },
    });
    return response.data; // Returning fetched data
  } catch (error) {
    console.error("Error fetching orders:", error);
    return [];
  }
}

async function putData(url: string, body: Record<string, string>){
  try {
    const response = await httpfetch.put(url, body,  {
      auth: {
        username: uname,
        password: pass,
      },
    });
    return response.data; // Returning fetched data
  } catch (error) {
    console.error("Error fetching orders:", error);
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
