import axios, { AxiosInstance } from "axios";

const httpfetch: AxiosInstance = axios.create({
  baseURL: "https://localhost/api",
  timeout: 2000,
  headers: {
    "Content-Type": "application/json", // Default content type
    Accept: "application/json", // Default accept header
  },
});

const getToken = () => {
  return localStorage.getItem("token") || "";
};

httpfetch.interceptors.request.use(
  (config) => {
    const token = getToken();
    const auth = token ? `Bearer ${token}` : "";
    config.headers["Authorization"] = auth;
    return config;
  },
  (error) => Promise.reject(error),
);

export default httpfetch;
