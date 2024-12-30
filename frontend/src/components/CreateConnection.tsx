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
import { Box, Button, ButtonGroup, FormHelperText } from "@mui/joy";
import { useAlert } from "../providers/UseAlert"
import { AxiosError, AxiosResponse } from 'axios';
import CheckCircleOutlineTwoToneIcon from '@mui/icons-material/CheckCircleOutlineTwoTone';
import ErrorTwoToneIcon from '@mui/icons-material/ErrorTwoTone';
import { colors } from "@mui/material";

type ConnectionSuccessType = null | boolean

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
  } catch (error: unknown) {
    return Promise.reject(error);
  }
};

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
    return Promise.reject(error);
  }
};

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
  const [formErrors, setFormErrors] = useState({
    name: "",
    description: ""
  })

  const fetchConnectionTypes = async () => {
    const data = await fetchRows();
    setRows(data);
  };

  useEffect(()=>{
    fetchConnectionTypes();
  }, [])

  useEffect(() => {
    if (selectedConnectionType !== "all") {
      const fetchCredentials = async () => {
        const data = await fetchFormFields(selectedConnectionType);
        setCredentialsForm(data);
        const credentialsFormKeys = Object.keys(data)
        if(credentialsFormKeys.length > 0){
          const errorsMap = credentialsFormKeys.reduce((acc, key) => {
              acc[key] = ""
            return acc
          }, {})
          setFormErrors(errorsMap)
        }
      };
      fetchCredentials();
    }
  }, [selectedConnectionType]);

  useEffect(() => {
    const handleFormChange = () => {
      if (!formRef.current) return;
      setIsFormValid(isFormFilled());
      setConnectionsSuccess(null);
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
    const isRedisConnection = selectedConnectionType === "REDIS";

  
    return Object.entries(data).every(([key, value]) => {
      // Skip validation for database and user fields if it's a Redis connection
      if (isRedisConnection && (key === "database" || key === "user")) {
        return true;
      }
      
      return value.toString().trim() !== "";
    });
  };

  const getConnectionData = (): ConnectionDataRequest | null => {
    if (!formRef.current) return null;
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
    const connectionData = getConnectionData();
    if (connectionData) {
      try {
        await saveConnection(connectionData, test);
        if (test) {
          setConnectionsSuccess(true);
        }
      } catch (error: unknown) {
        if(error instanceof(AxiosError)){
          if (error?.response?.data != undefined){
            if ("credentials" in error.response.data){
              const errorsList = error.response.data.credentials
             Object.keys(error.response.data.credentials).forEach((key) => {
              setFormErrors((prevErrors) => ({
                ...prevErrors,
                [key]: errorsList[key],
              }));
            });
            }
          }
        }
        setConnectionsSuccess(false);
        showAlert("Error submittion connection");
      }
    } else {
      showAlert("Some connection credentials are missing");
    }
  };

  const renderCredentialFields = useMemo(() => {
    const form = Object.entries(credentialsForm).map(([key, field]) => {
      const title = key.charAt(0).toUpperCase() + key.slice(1);
      return (
        <FormControl key={key} error={formErrors[key]}>
          <FormLabel>{title}</FormLabel>
          <Input
            name={key}
            type={field.type === "CharField" ? "text" : "number"}
            required={field.required}
          />
          <FormHelperText>{key in formErrors ? formErrors[key]: ""}</FormHelperText>
        </FormControl>
      );
    });
    return form
  }, [credentialsForm, formErrors]);

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
            <ButtonGroup>
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
            </ButtonGroup>
          )}
        </Box>
      </Sheet>
    </React.Fragment>
  );
};

export default CreateConnection;
