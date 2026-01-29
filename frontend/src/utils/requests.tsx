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
  join?: {
    table: string;
    left_column: string;
    right_column: string;
    join_type?: JoinType;
    columns?: string[];
  };
}) => {
  try {
    const response = await httpfetch.post(`datasource/export/${connectionId}/`, exportParams);
    return response.data;
  } catch (error) {
    console.error("Error exporting table:", error);
    throw error;
  }
}

// Segmentation API functions
export const getDatasetSegmentation = async (datasetId: string) => {
  return getData(`datasets/${datasetId}/segmentation/`);
}

export const getSegments = async (datasetId?: string) => {
  const url = datasetId ? `segments/?dataset_id=${datasetId}` : 'segments/';

  // Fetch all pages if paginated
  try {
    const allSegments = [];
    let nextUrl: string | null = url;
    const baseURL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/";

    while (nextUrl) {
      const response = await httpfetch.get(nextUrl);
      const data = response.data;

      // Handle paginated response
      if (data.results) {
        allSegments.push(...data.results);
        // Strip base URL from next URL if it's absolute
        if (data.next) {
          nextUrl = data.next.replace(baseURL, '');
        } else {
          nextUrl = null;
        }
      } else {
        // Non-paginated response
        return data;
      }
    }

    return allSegments;
  } catch (error) {
    console.error("Error fetching segments:", error);
    throw error;
  }
}

export const createSegment = async (segmentData: {
  dataset: string;
  name: string;
  description?: string;
  color?: string;
}) => {
  try {
    const response = await httpfetch.post('segments/', segmentData);
    return response.data;
  } catch (error) {
    console.error("Error creating segment:", error);
    throw error;
  }
}

export const updateSegment = async (segmentId: string, segmentData: Partial<{
  name: string;
  description: string;
  color: string;
}>) => {
  try {
    const response = await httpfetch.patch(`segments/${segmentId}/`, segmentData);
    return response.data;
  } catch (error) {
    console.error("Error updating segment:", error);
    throw error;
  }
}

export const deleteSegment = async (segmentId: string) => {
  try {
    const response = await httpfetch.delete(`segments/${segmentId}/`);
    return response.data;
  } catch (error) {
    console.error("Error deleting segment:", error);
    throw error;
  }
}

export const getSegmentLabels = async (segmentId?: string, datasetId?: string, rowIndices?: number[]) => {
  let url = 'segment-labels/';
  const params = [];
  if (segmentId) params.push(`segment_id=${segmentId}`);
  if (datasetId) params.push(`dataset_id=${datasetId}`);
  // If specific row indices are provided, add to query params
  if (rowIndices && rowIndices.length > 0) {
    params.push(`row_indices=${rowIndices.join(',')}`);
  }
  if (params.length > 0) url += '?' + params.join('&');

  // When fetching by row_indices, we typically get a small result set that doesn't need pagination
  // Just do a single request for efficiency
  try {
    const response = await httpfetch.get(url);
    const data = response.data;

    // Handle paginated response
    if (data.results) {
      return data.results;
    } else {
      // Non-paginated response
      return data;
    }
  } catch (error) {
    console.error("Error fetching segment labels:", error);
    throw error;
  }
}

export const bulkAssignRows = async (assignmentData: {
  segment_id: string;
  row_indices: number[];
  assignment_method?: 'manual' | 'rule' | 'ml';
  confidence?: number;
  notes?: string;
}) => {
  try {
    const response = await httpfetch.post('segment-labels/bulk_assign/', assignmentData);
    return response.data;
  } catch (error) {
    console.error("Error bulk assigning rows:", error);
    throw error;
  }
}

export const bulkDeleteLabels = async (segment_id: string, row_indices: number[]) => {
  try {
    const response = await httpfetch.post('segment-labels/bulk_delete/', {
      segment_id,
      row_indices
    });
    return response.data;
  } catch (error) {
    console.error("Error bulk deleting labels:", error);
    throw error;
  }
}

export const getSegmentStatistics = async (segmentId: string) => {
  return getData(`segments/${segmentId}/statistics/`);
}

export const exportSegment = async (segmentId: string) => {
  try {
    const response = await httpfetch.post(`segments/${segmentId}/export/`);
    return response.data;
  } catch (error) {
    console.error("Error exporting segment:", error);
    throw error;
  }
}

// Dataset Join API functions
export const getDatasetColumns = async (datasetId: string) => {
  try {
    const response = await httpfetch.get(`datasets/${datasetId}/columns/`);
    return response.data;
  } catch (error) {
    console.error("Error fetching dataset columns:", error);
    throw error;
  }
}

export type JoinType = 'inner' | 'left' | 'right' | 'outer';

export const joinDatasets = async (joinParams: {
  left_dataset_id: string;
  right_dataset_id: string;
  left_key_column: string;
  right_key_column: string;
  join_type?: JoinType;
  result_name: string;
  result_description?: string;
}) => {
  try {
    const response = await httpfetch.post('datasets/join/', joinParams);
    return response.data;
  } catch (error) {
    console.error("Error joining datasets:", error);
    throw error;
  }
}

export const exportTableWithJoin = async (connectionId: string, exportParams: {
  schema: string;
  table: string;
  columns?: string[];
  filters?: string;
  limit?: number;
  dataset_name?: string;
  dataset_description?: string;
  join?: {
    table: string;
    left_column: string;
    right_column: string;
    join_type?: JoinType;
    columns?: string[];
  };
}) => {
  try {
    const response = await httpfetch.post(`datasource/export/${connectionId}/`, exportParams);
    return response.data;
  } catch (error) {
    console.error("Error exporting table with join:", error);
    throw error;
  }
}

// Auto-clustering API functions
export const autoClusterDataset = async (datasetId: string, params: {
  algorithm: 'kmeans' | 'dbscan' | 'hierarchical' | 'gaussian_mixture' | 'mean_shift';
  config?: {
    n_clusters?: number;
    eps?: number;
    min_samples?: number;
    linkage?: string;
    bandwidth?: number;
    covariance_type?: string;
  };
  feature_columns?: string[];
  create_segments?: boolean;
  segment_prefix?: string;
}) => {
  try {
    const response = await httpfetch.post(`datasets/${datasetId}/auto-cluster/`, params);
    return response.data;
  } catch (error) {
    console.error("Error auto-clustering dataset:", error);
    throw error;
  }
}

export const getOptimalClusters = async (datasetId: string, params?: {
  feature_columns?: string[];
  max_clusters?: number;
  method?: 'elbow' | 'silhouette';
}) => {
  try {
    const response = await httpfetch.post(`datasets/${datasetId}/optimal-clusters/`, params || {});
    return response.data;
  } catch (error) {
    console.error("Error determining optimal clusters:", error);
    throw error;
  }
}

export interface GpuInfo {
  cuda_available: boolean;
  gpu_count: number;
  gpus: {
    index: number;
    name: string;
    total_memory_gb: number;
    major: number;
    minor: number;
    multi_processor_count: number;
  }[];
  cuda_version: string | null;
  cudnn_version: string | null;
  error?: string;
}

export const getGpuInfo = async (): Promise<GpuInfo> => {
  try {
    const response = await httpfetch.get('system/gpu-info/');
    return response.data;
  } catch (error) {
    console.error("Error fetching GPU info:", error);
    return {
      cuda_available: false,
      gpu_count: 0,
      gpus: [],
      cuda_version: null,
      cudnn_version: null,
      error: 'Failed to fetch GPU info'
    };
  }
}
