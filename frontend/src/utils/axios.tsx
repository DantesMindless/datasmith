
const baseURL = import.meta.env.VITE_API_URL || "http://localhost:8000/api/";

const getAccessToken = () => {
  return localStorage.getItem("access_token") || "";
};

const getRefreshToken = () => {
  return localStorage.getItem("refresh_token") || "";
};

const setTokens = (accessToken: string, refreshToken: string) => {
  localStorage.setItem("access_token", accessToken);
  localStorage.setItem("refresh_token", refreshToken);
};

const clearTokens = () => {
  localStorage.removeItem("access_token");
  localStorage.removeItem("refresh_token");
  localStorage.removeItem("token"); // Clear old token too
};

interface RequestOptions extends RequestInit {
  data?: any;
}

const refreshAccessToken = async (): Promise<string | null> => {
  const refreshToken = getRefreshToken();
  if (!refreshToken) return null;
  
  try {
    const response = await fetch(`${baseURL}refresh/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ refresh: refreshToken }),
      credentials: 'omit',
    });
    
    if (response.ok) {
      const data = await response.json();
      localStorage.setItem("access_token", data.access);
      return data.access;
    }
  } catch (error) {
    console.error("Token refresh failed:", error);
  }
  
  // If refresh fails, clear tokens
  clearTokens();
  window.location.href = '/'; // Redirect to login
  return null;
};

const httpfetch = {
  async request(url: string, options: RequestOptions = {}, isRetry: boolean = false) {
    const accessToken = getAccessToken();
    
    const headers: HeadersInit = {
      "Content-Type": "application/json",
      "Accept": "application/json",
      ...(options.headers || {}),
    };
    
    if (accessToken) {
      headers["Authorization"] = `Bearer ${accessToken}`;
    }
    
    console.log("Fetch request - Access token:", accessToken);
    console.log("Fetch request - Auth header:", headers["Authorization"]);
    console.log("Fetch request - URL:", `${baseURL}${url}`);
    
    const config: RequestInit = {
      ...options,
      headers,
      credentials: 'omit', // Don't send browser stored credentials
    };
    
    // Handle request body
    if (options.data) {
      if (options.data instanceof FormData) {
        // For FormData, remove content-type to let browser set it
        delete (headers as any)["Content-Type"];
        config.body = options.data;
      } else {
        config.body = JSON.stringify(options.data);
      }
    }
    
    const response = await fetch(`${baseURL}${url}`, config);
    
    // If 401 and not already retrying, try to refresh token
    if (response.status === 401 && !isRetry && accessToken) {
      console.log("Token expired, attempting refresh...");
      const newToken = await refreshAccessToken();
      if (newToken) {
        // Retry the request with new token
        return this.request(url, options, true);
      }
    }
    
    if (!response.ok) {
      let errorData;
      try {
        errorData = await response.text();
        // Try to parse as JSON if possible
        try {
          errorData = JSON.parse(errorData);
        } catch {
          // Keep as text if not valid JSON
        }
      } catch {
        errorData = null;
      }

      const error = new Error(`Request failed with status ${response.status}`);
      (error as any).response = {
        status: response.status,
        data: errorData,
      };
      throw error;
    }

    // For successful responses, handle different content types
    const contentType = response.headers.get('Content-Type') || '';
    let data;

    if (response.status === 204 || response.status === 205) {
      // No content responses
      data = null;
    } else if (contentType.includes('application/json')) {
      try {
        data = await response.json();
      } catch {
        data = null;
      }
    } else {
      data = await response.text();
    }

    return { data, status: response.status, statusText: response.statusText };
  },
  
  get(url: string, options?: RequestOptions) {
    return this.request(url, { ...options, method: 'GET' });
  },
  
  post(url: string, data?: any, options?: RequestOptions) {
    return this.request(url, { ...options, method: 'POST', data });
  },
  
  put(url: string, data?: any, options?: RequestOptions) {
    return this.request(url, { ...options, method: 'PUT', data });
  },
  
  delete(url: string, options?: RequestOptions) {
    return this.request(url, { ...options, method: 'DELETE' });
  }
};

export default httpfetch;
export { setTokens, clearTokens, getAccessToken, getRefreshToken };
