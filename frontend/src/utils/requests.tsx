import httpfetch from './axios';
import { UUIDTypes } from 'uuid';

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

export const getConnections = async () => {
  return getData("datasource/")
};

export const getSchemaTablesList = async (uuid : UUIDTypes, tableName:string) => {
  return getData(`datasource-schema-metadata/${uuid}/${tableName}/`)
};

export const getTablesList = async (uuid : UUIDTypes) => {
  return getData(`datasource-metadata/tables/${uuid}/`)
};

export const getDatabasesList = async (uuid : UUIDTypes) => {
  return getData(`datasource-metadata/schemas/${uuid}/`)
};

export const getConnectionTypes = async () => {
  return getData("datasource/connection-types/")
};
