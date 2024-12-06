import axios from 'axios';

const axiosInstance = axios.create({
  baseURL: 'http://localhost:8000/api', // Set the base URL
  timeout: 5000, // Optional: Set a timeout for requests (in milliseconds)
  headers: {
    'Content-Type': 'application/json', // Default content type
    Accept: 'application/json', // Default accept header
  },
});
export default axiosInstance;