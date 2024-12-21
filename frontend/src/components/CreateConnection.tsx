import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useMemo,
} from "react";
import FormControl from "@mui/joy/FormControl";
import FormLabel from "@mui/joy/FormLabel";
import Input from "@mui/joy/Input";
import Select from "@mui/joy/Select";
import Option from "@mui/joy/Option";
import Sheet from "@mui/joy/Sheet";
import httpfetch from "../utils/axios";
import { Box, Button } from "@mui/joy";
import AlertSnackbar from "./helpers/AlertBanner"
import { useAlert } from "../providers/UseAlert"
import { AxiosResponse } from 'axios';
import CheckCircleOutlineTwoToneIcon from '@mui/icons-material/CheckCircleOutlineTwoTone';
import ErrorTwoToneIcon from '@mui/icons-material/ErrorTwoTone';

type ConnectionSuccessType = null | boolean

// Types for API responses
interface ConnectionType {
  value: string;
  title: string;
}

interface ConnectionDataRequest {
  type: string;
  name: string;
  description: string;
  credentials: Record<string, string | number>;
}

interface CredentialField {
  type: string;
  required: boolean;
}

type CredentialsForm = Record<string, CredentialField>;

// Function to fetch connection types
const fetchRows = async (): Promise<ConnectionType[]> => {
  try {
    const response: AxiosResponse = await httpfetch.get("datasource/connection-types/", {
      auth: { username: "u@u.com", password: "password" },
    });
    return response.data;
  } catch (error) {
    console.error("Error fetching connection types:", error);
    return [];
  }
};

// Function to fetch form fields for selected connection type
const fetchFormFields = async (
  connectionType: string,
): Promise<CredentialsForm> => {
  try {
    const response: AxiosResponse = await httpfetch.get(
      `datasource/connection-type-form/${connectionType}`,
      {
        auth: { username: "u@u.com", password: "password" },
      },
    );
    return response.data;
  } catch (error) {
    console.error("Error fetching form fields:", error);
    return {};
  }
};

// Function to save connection data
const saveConnection = async (
  data: ConnectionDataRequest,
  test: boolean,
): Promise<AxiosResponse> => {
  try {
    const response: AxiosResponse = await httpfetch.post(`datasource/${test ? "test/" : ""}`, data, {
      auth: { username: "u@u.com", password: "password" },
    });
    return response.data;
  } catch (error) {
    return Promise.reject(error);
  }
};
const CreateConnection: React.FC = () => {
  const { showAlert } = useAlert();
  const formRef = useRef<HTMLFormElement>(null);
  const [rows, setRows] = useState<ConnectionType[]>([]);
  const [selectedConnectionType, setSelectedConnectionType] = useState<string>("all");
  const [credentialsForm, setCredentialsForm] = useState<CredentialsForm>({});
  const [connectionSuccess, setConnectionsSuccess] = useState<ConnectionSuccessType>(null);
  const [isFormValid, setIsFormValid] = useState(false); // State to track form validity

  useEffect(() => {
    // Fetch form fields when a new connection type is selected
    const fetchConnectionTypes = async () => {
      const data = await fetchRows();
      setRows(data);
    };

    fetchConnectionTypes();
    if (selectedConnectionType !== "all") {
      const fetchCredentials = async () => {
        const data = await fetchFormFields(selectedConnectionType);
        setCredentialsForm(data);
      };

      fetchCredentials();
    }
  }, [selectedConnectionType]);

  useEffect(() => {
    // Check form validity whenever the form changes
    const handleFormChange = () => {
      if (!formRef.current) return;
      setIsFormValid(isFormFilled());
    };

    const form = formRef.current;
    if (form) {
      form.addEventListener("input", handleFormChange);
    }

    return () => {
      if (form) {
        form.removeEventListener("input", handleFormChange);
      }
    };
  }, [credentialsForm, selectedConnectionType]);

  const handleChange = useCallback(
    (event: React.SyntheticEvent | null, newValue: string | null) => {
      if (newValue && newValue !== "all") {
        setSelectedConnectionType(newValue);
      }
    },
    []
  );

  const isFormFilled = () => {
    if (!formRef.current) return false;
    const formData = new FormData(formRef.current);
    const data = Object.fromEntries(formData.entries());
    return Object.values(data).every((value) => value.toString().trim() !== "");
  };

  const getConnectionData: ConnectionDataRequest = () => {
    if (!formRef.current) return;

    const formData = new FormData(formRef.current);
    const data = Object.fromEntries(formData.entries());

    return {
      type: selectedConnectionType,
      name: data.name as string,
      description: data.description as string,
      credentials: Object.keys(credentialsForm).reduce((acc, key) => {
        acc[key] = data[key] as string;
        return acc;
      }, {} as Record<string, string>),
    };
  };

  const handleSubmit = async (e: React.FormEvent, test: boolean = false) => {
    e.preventDefault();
    if (getConnectionData() != null) {
      try {
        await saveConnection(getConnectionData(), test);
        if (test) {
          setConnectionsSuccess(true);
        }
      } catch (error: unknown) {
        setConnectionsSuccess(false);
        showAlert(error.response.data);
      }
    } else {
      showAlert("Some connection credentials are missing");
    }
  };

  const renderCredentialFields = useMemo(() => {
    return Object.entries(credentialsForm).map(([key, field]) => {
      const title = key.charAt(0).toUpperCase() + key.slice(1);
      return (
        <FormControl key={key}>
          <FormLabel>{title}</FormLabel>
          <Input
            name={key}
            type={field.type === "CharField" ? "text" : "number"}
            required={field.required}
          />
        </FormControl>
      );
    });
  }, [credentialsForm]);

  return (
    <React.Fragment>
      <Sheet>
        <Box component="form" ref={formRef}>
          <FormControl size="sm">
            <FormLabel>Select Connection Type</FormLabel>
            <Select
              name="type"
              size="sm"
              placeholder="Select Connection"
              onChange={handleChange}
              value={selectedConnectionType}
            >
              <Option value="all">Select Connection</Option>
              {rows.map((item) => (
                <Option key={item.value} value={item.value}>
                  {item.title}
                </Option>
              ))}
            </Select>
          </FormControl>
          <FormControl>
            <FormLabel>Connection Title</FormLabel>
            <Input name="name" required />
          </FormControl>
          <FormControl>
            <FormLabel>Description</FormLabel>
            <Input name="description" required />
          </FormControl>
          {renderCredentialFields}
          {selectedConnectionType !== "all" && (
            <div>
              <Button
                type="submit"
                variant="outlined"
                color="primary"
                onClick={handleSubmit}
                disabled={!isFormValid}
              >
                Submit
              </Button>
              <Button
                type="button"
                variant="outlined"
                color={`${isFormValid ? "primary" : "success"}`}
                onClick={(e) => handleSubmit(e, true)}
                disabled={!isFormValid}
              >
                Test Connection
                {connectionSuccess === true && (
                  <CheckCircleOutlineTwoToneIcon color="success" />
                )}
                {connectionSuccess === false && (
                  <ErrorTwoToneIcon color="warning" />
                )}
              </Button>
            </div>
          )}
        </Box>
      </Sheet>
    </React.Fragment>
  );
};

export default CreateConnection;
