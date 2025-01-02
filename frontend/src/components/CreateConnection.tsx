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
import { useAppContext } from '../providers/useAppContext';
import { AxiosError, AxiosResponse } from 'axios';
import CheckCircleOutlineTwoToneIcon from '@mui/icons-material/CheckCircleOutlineTwoTone';
import ErrorTwoToneIcon from '@mui/icons-material/ErrorTwoTone';
import { getConnectionTypes } from '../utils/requests'


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
  initial: string | number;
}

type CredentialsForm = Record<string, CredentialField>;



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
  const { showAlert } = useAppContext();
  const formRef = useRef<HTMLFormElement>(null);
  const [rows, setRows] = useState<ConnectionType[]>([]);
  const [selectedConnectionType, setSelectedConnectionType] = useState<string>("all");
  const [credentialsForm, setCredentialsForm] = useState<CredentialsForm>({});
  const [connectionSuccess, setConnectionsSuccess] = useState<ConnectionSuccessType>(null);
  const [isFormValid, setIsFormValid] = useState(false);
  const [formErrors, setFormErrors] = useState({
    name: "",
    description: ""
  });
  // Add a key to force re-render of credential fields
  const [formKey, setFormKey] = useState(0);

  const fetchConnectionTypes = async () => {
    const data = await getConnectionTypes();
    setRows(data);
  };

  useEffect(()=>{
    fetchConnectionTypes();
  }, []);

  useEffect(() => {
    if (selectedConnectionType !== "all") {
      const fetchCredentials = async () => {
        const data = await fetchFormFields(selectedConnectionType);
        setCredentialsForm(data);
        setFormKey(prev => prev + 1); // Increment key to force re-render
        
        const credentialsFormKeys = Object.keys(data);
        if(credentialsFormKeys.length > 0){
          const errorsMap = credentialsFormKeys.reduce((acc, key) => {
            acc[key] = "";
            return acc;
          }, {});
          setFormErrors(errorsMap);
        }

        // Reset form if it exists
        if (formRef.current) {
          formRef.current.reset();
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
      if (newValue) {
        setSelectedConnectionType(newValue);
        setConnectionsSuccess(null);
        if (formRef.current) {
          formRef.current.reset();
        }
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

  const getConnectionData = (): ConnectionDataRequest | null => {
    if (!formRef.current) return null;
    const formData = new FormData(formRef.current);
    const data = Object.fromEntries(formData.entries());

    return {
      type: selectedConnectionType,
      name: data.name as string,
      description: data.description as string,
      credentials: Object.keys(credentialsForm).reduce((acc, key) => {
        const value = data[key] as string;
        // Convert to number if the field type is IntegerField
        acc[key] = credentialsForm[key].type === "IntegerField" ? Number(value) : value;
        return acc;
      }, {} as Record<string, string | number>),
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
              const errorsList = error.response.data.credentials;
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
        showAlert("Error submitting connection");
      }
    } else {
      showAlert("Some connection credentials are missing");
    }
  };

  const renderCredentialFields = useMemo(() => {
    const form = Object.entries(credentialsForm).map(([key, field]) => {
      const title = key.charAt(0).toUpperCase() + key.slice(1);
      return (
        <FormControl key={`${key}-${formKey}`} error={formErrors[key]}>
          <FormLabel>{title}</FormLabel>
          <Input
            name={key}
            type={field.type === "CharField" ? "text" : "number"}
            required={field.required}
            defaultValue={field.initial}
          />
          <FormHelperText>{key in formErrors ? formErrors[key]: ""}</FormHelperText>
        </FormControl>
      );
    });
    return form;
  }, [credentialsForm, formErrors, formKey]);

  return (
    <React.Fragment>
      <Sheet>
        <Box component="form" ref={formRef} key={formKey}>
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
