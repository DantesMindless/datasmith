import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  Box,
  Button,
  ButtonGroup,
  FormControl,
  MenuItem,
  Select,
  TextField,
} from "@mui/material";
import CheckCircleOutlineTwoToneIcon from "@mui/icons-material/CheckCircleOutlineTwoTone";
import ErrorTwoToneIcon from "@mui/icons-material/ErrorTwoTone";
import { useAppContext } from "../providers/useAppContext";
import httpfetch from "../utils/axios";
import { AxiosError, AxiosResponse } from "axios";
import { getConnectionTypes } from "../utils/requests";

type ConnectionSuccessType = null | boolean;

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

const fetchFormFields = async (
  connectionType: string
): Promise<CredentialsForm> => {
  try {
    const response: AxiosResponse = await httpfetch.get(
      `datasource/connection-type-form/${connectionType}`,
      {
        auth: { username: "u@u.com", password: "password" },
      }
    );
    return response.data;
  } catch (error) {
    return Promise.reject(error);
  }
};

const saveConnection = async (
  data: ConnectionDataRequest,
  test: boolean
): Promise<AxiosResponse> => {
  try {
    const response: AxiosResponse = await httpfetch.post(
      `datasource/${test ? "test/" : ""}`,
      data,
      {
        auth: { username: "u@u.com", password: "password" },
      }
    );
    return response.data;
  } catch (error) {
    return Promise.reject(error);
  }
};

const CreateConnection: React.FC = () => {
  const { showAlert, showInfo, updateConnections } = useAppContext();
  const formRef = useRef<HTMLFormElement>(null);
  const [rows, setRows] = useState<ConnectionType[]>([]);
  const [selectedConnectionType, setSelectedConnectionType] = useState<string>(
    "all"
  );
  const [credentialsForm, setCredentialsForm] = useState<CredentialsForm>({});
  const [connectionSuccess, setConnectionsSuccess] = useState<ConnectionSuccessType>(null);
  const [isFormValid, setIsFormValid] = useState(false);
  const [formErrors, setFormErrors] = useState({ name: "", description: "" });

  const fetchConnectionTypes = async () => {
    const data = await getConnectionTypes();
    setRows(data);
  };

  useEffect(() => {
    fetchConnectionTypes();
  }, []);

  useEffect(() => {
    if (selectedConnectionType !== "all") {
      const fetchCredentials = async () => {
        const data = await fetchFormFields(selectedConnectionType);
        setCredentialsForm(data);
        const credentialsFormKeys = Object.keys(data);
        if (credentialsFormKeys.length > 0) {
          const errorsMap = credentialsFormKeys.reduce((acc, key) => {
            acc[key] = "";
            return acc;
          }, {});
          setFormErrors(errorsMap);
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
    (event: React.ChangeEvent<{ value: unknown }>) => {
      const newValue = event.target.value as string;
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
        showInfo("Connection saved successfully");
        await saveConnection(connectionData, test);
        if (test) {
          setConnectionsSuccess(true);
        } else {
          // Update the connections list after successful save
          await updateConnections();
          showInfo("Connection saved successfully");
        }
      } catch (error: unknown) {
        if (error instanceof AxiosError && error?.response?.data?.credentials) {
          const errorsList = error.response.data.credentials;
          Object.keys(errorsList).forEach((key) => {
            setFormErrors((prevErrors) => ({
              ...prevErrors,
              [key]: errorsList[key],
            }));
          });
        }
        setConnectionsSuccess(false);
        showAlert("Error submitting connection");
      }
    } else {
      showAlert("Some connection credentials are missing");
    }
  };

  const renderCredentialFields = useMemo(() => {
    return Object.entries(credentialsForm).map(([key, field]) => {
      const title = key.charAt(0).toUpperCase() + key.slice(1);
      return (
        <FormControl key={key} error={!!formErrors[key]} sx={{my: 1}}>
          <TextField
            label={key}
            id="filled-size-normal"
            name={key}
            type={field.type === "CharField" ? "text" : "number"}
            required={field.required}
            helperText={formErrors[key]}
          />
        </FormControl>
      );
    });
  }, [credentialsForm, formErrors]);

  return (
    <Box component="form" ref={formRef} sx={{ display: 'flex', flexDirection: 'column',  p: 3, bgcolor: "background.paper" }}>
      <FormControl fullWidth margin="normal" sx={{my: 1}}>
        <Select
          id="filled-size-normal"
          name="type"
          value={selectedConnectionType}
          onChange={handleChange}
        >
          <MenuItem value="all">Select Connection</MenuItem>
          {rows.map((item) => (
            <MenuItem key={item.value} value={item.value}>
              {item.title}
            </MenuItem>
          ))}
        </Select>
      </FormControl>
      <FormControl fullWidth margin="normal" sx={{my: 1}}>
      <TextField
        id="filled-size-normal"
        label="Connection Title"
        name="name"
        required
      />
      </FormControl>
      <FormControl fullWidth margin="normal" sx={{my: 1}}>
      <TextField
        id="filled-size-normal"
        label="Description"
        name="description"
        required
      />
      </FormControl>

      {renderCredentialFields}
      {selectedConnectionType !== "all" && (
        <ButtonGroup fullWidth>
          <Button
            variant="contained"
            color="primary"
            onClick={(e) => handleSubmit(e)}
            disabled={!isFormValid}
          >
            Submit
          </Button>
          <Button
            variant="outlined"
            color={connectionSuccess ? "success" : "warning"}
            onClick={(e) => handleSubmit(e, true)}
            disabled={!isFormValid}
          >
            Test Connection
            {connectionSuccess === true && (
              <CheckCircleOutlineTwoToneIcon sx={{ ml: 1 }} />
            )}
            {connectionSuccess === false && (
              <ErrorTwoToneIcon sx={{ ml: 1 }} />
            )}
          </Button>
        </ButtonGroup>
      )}
    </Box>
  );
};

export default CreateConnection;
