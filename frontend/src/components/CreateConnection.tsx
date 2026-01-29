import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  Box,
  Button,
  ButtonGroup,
  FormControl,
  MenuItem,
  Select,
  TextField,
  Typography,
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

export interface EditConnectionData {
  id: string;
  name: string;
  description?: string;
  type: string;
  credentials: Record<string, string | number>;
}

interface CreateConnectionProps {
  editData?: EditConnectionData | null;
  onClose?: () => void;
}

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

const updateConnection = async (
  id: string,
  data: ConnectionDataRequest
): Promise<AxiosResponse> => {
  try {
    const response: AxiosResponse = await httpfetch.patch(
      `datasource/detail/${id}/`,
      data
    );
    return response.data;
  } catch (error) {
    return Promise.reject(error);
  }
};

const CreateConnection: React.FC<CreateConnectionProps> = ({ editData, onClose }) => {
  const { showAlert, showInfo, updateConnections } = useAppContext();
  const formRef = useRef<HTMLFormElement>(null);
  const [rows, setRows] = useState<ConnectionType[]>([]);
  const [selectedConnectionType, setSelectedConnectionType] = useState<string>(
    editData?.type || "all"
  );
  const [credentialsForm, setCredentialsForm] = useState<CredentialsForm>({});
  const [connectionSuccess, setConnectionsSuccess] = useState<ConnectionSuccessType>(null);
  const [isFormValid, setIsFormValid] = useState(false);
  const [formErrors, setFormErrors] = useState<Record<string, string>>({ name: "", description: "" });

  // Form values for controlled inputs
  const [formValues, setFormValues] = useState<Record<string, string>>({
    name: editData?.name || "",
    description: editData?.description || "",
  });

  const isEditMode = !!editData;

  const fetchConnectionTypes = async () => {
    const data = await getConnectionTypes();
    setRows(data);
  };

  useEffect(() => {
    fetchConnectionTypes();
  }, []);

  // Initialize form when editing
  useEffect(() => {
    if (editData) {
      setSelectedConnectionType(editData.type);
      setFormValues({
        name: editData.name,
        description: editData.description || "",
        ...editData.credentials,
      });
    }
  }, [editData]);

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
          }, {} as Record<string, string>);
          setFormErrors(prev => ({ ...prev, ...errorsMap }));

          // If editing, populate credential values
          if (editData?.credentials) {
            setFormValues(prev => ({
              ...prev,
              ...Object.fromEntries(
                Object.entries(editData.credentials).map(([k, v]) => [k, String(v)])
              ),
            }));
          }
        }
      };
      fetchCredentials();
    }
  }, [selectedConnectionType, editData]);

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
  }, [credentialsForm, selectedConnectionType, formValues]);

  const handleChange = useCallback(
    (event: React.ChangeEvent<{ value: unknown }>) => {
      const newValue = event.target.value as string;
      if (newValue && newValue !== "all") {
        setSelectedConnectionType(newValue);
      }
    },
    []
  );

  const handleInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const { name, value } = event.target;
      setFormValues(prev => ({ ...prev, [name]: value }));
    },
    []
  );

  const isFormFilled = () => {
    if (!formRef.current) return false;
    const formData = new FormData(formRef.current);
    const data = Object.fromEntries(formData.entries());

    return Object.entries(data).every(([key, value]) => {
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
        if (isEditMode && editData && !test) {
          // Update existing connection
          await updateConnection(editData.id, connectionData);
          await updateConnections();
          showInfo("Connection updated successfully!");
          onClose?.();
        } else if (test) {
          await saveConnection(connectionData, true);
          setConnectionsSuccess(true);
          showInfo("Connection test successful!");
        } else {
          await saveConnection(connectionData, false);
          await updateConnections();
          showInfo("Connection saved successfully!");

          // Reset form after successful save
          if (formRef.current) {
            formRef.current.reset();
          }
          setSelectedConnectionType("all");
          setCredentialsForm({});
          setConnectionsSuccess(null);
          setIsFormValid(false);
          setFormValues({ name: "", description: "" });
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
        const action = isEditMode ? "updating" : test ? "testing" : "saving";
        showAlert(`Error ${action} connection`);
      }
    } else {
      showAlert("Some connection credentials are missing");
    }
  };

  const renderCredentialFields = useMemo(() => {
    return Object.entries(credentialsForm).map(([key, field]) => {
      return (
        <FormControl key={key} error={!!formErrors[key]} sx={{my: 1}}>
          <TextField
            label={key}
            id={`field-${key}`}
            name={key}
            type={field.type === "CharField" ? "text" : "number"}
            required={field.required}
            helperText={formErrors[key]}
            value={formValues[key] || ""}
            onChange={handleInputChange}
          />
        </FormControl>
      );
    });
  }, [credentialsForm, formErrors, formValues, handleInputChange]);

  return (
    <Box component="form" ref={formRef} sx={{ display: 'flex', flexDirection: 'column', p: 3, bgcolor: "background.paper" }}>
      {isEditMode && (
        <Typography variant="h6" sx={{ mb: 2 }}>
          Edit Connection
        </Typography>
      )}
      <FormControl fullWidth margin="normal" sx={{my: 1}}>
        <Select
          id="filled-size-normal"
          name="type"
          value={selectedConnectionType}
          onChange={handleChange}
          disabled={isEditMode}
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
        value={formValues.name || ""}
        onChange={handleInputChange}
      />
      </FormControl>
      <FormControl fullWidth margin="normal" sx={{my: 1}}>
      <TextField
        id="filled-size-normal"
        label="Description"
        name="description"
        required
        value={formValues.description || ""}
        onChange={handleInputChange}
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
            {isEditMode ? "Update" : "Submit"}
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
          {isEditMode && onClose && (
            <Button
              variant="text"
              color="inherit"
              onClick={onClose}
            >
              Cancel
            </Button>
          )}
        </ButtonGroup>
      )}
    </Box>
  );
};

export default CreateConnection;
