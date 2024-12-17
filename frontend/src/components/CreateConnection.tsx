import * as React from "react";
import { useState, useEffect } from "react";
import FormControl from "@mui/joy/FormControl";
import FormLabel from "@mui/joy/FormLabel";
import Input from "@mui/joy/Input";
import Select from "@mui/joy/Select";
import Option from "@mui/joy/Option";
import Sheet from "@mui/joy/Sheet";
import httpfetch from "../utils/axios";
import TextField from "@mui/material/TextField";
import { Button } from "@mui/joy";

const uname = "u@u.com";
const pass = "password";

// Function to fetch orders
const fetchRows = async () => {
  try {
    const response = await httpfetch.get("datasource/connection-types/", {
      auth: {
        username: uname,
        password: pass,
      },
    });
    return response.data; // Returning fetched data
  } catch (error) {
    console.error("Error fetching orders:", error);
    return []; // Returning an empty array in case of error
  }
};

const fetchFormFields = async (connectionType: String) => {
  try {
    const response = await httpfetch.get(
      `datasource/connection-type-form/${connectionType}`,
      {
        auth: {
          username: uname,
          password: pass,
        },
      },
    );
    return response.data; // Returning fetched data
  } catch (error) {
    console.error("Error fetching orders:", error);
    return []; // Returning an empty array in case of error
  }
};

// Function to fetch orders
const saveConnection = async () => {
  try {
    const response = await httpfetch.post("datasource", {
      auth: {
        username: uname,
        password: pass,
      },
    });
    return response.data; // Returning fetched data
  } catch (error) {
    console.error("Error fetching orders:", error);
    return []; // Returning an empty array in case of error
  }
};

export default function CreateConnection() {
  const handleChange = (
    event: React.SyntheticEvent | null,
    newValue: string | null,
  ) => {
    if (newValue && newValue != "all") {
      setselectetConnectionType(newValue);
    }
  };

  const [rows, setRows] = useState([]);
  const [selectetConnectionType, setselectetConnectionType] = useState("all");
  const [credentialsForm, setCredentialsForm] = useState({});

  useEffect(() => {
    let isMounted = true;
    if (isMounted && rows.length === 0) {
      const fetchConnectionTypes = async () => {
        const data = await fetchRows();
        setRows(data);
      };
      fetchConnectionTypes();
    }
    if (selectetConnectionType != "all") {
      const fetchcredentialsForm = async () => {
        const data = await fetchFormFields(selectetConnectionType);
        setCredentialsForm(data);
      };
      fetchcredentialsForm();
    }
    return () => {
      isMounted = true;
    };
  }, [selectetConnectionType]);

  return (
    <React.Fragment>
      <Sheet>
        <FormControl size="sm">
          <FormLabel>Select Connection Type</FormLabel>
          <Select
            size="sm"
            placeholder="Select Connection"
            onChange={handleChange}
          >
            <Option value="all">Select Connection</Option>
            {[...rows].map((item) => (
              <Option value={item.value}>{item.title}</Option>
            ))}
          </Select>
          <FormLabel>Connection Title</FormLabel>
          <Input></Input>
          <FormLabel>Description</FormLabel>
          <Input required={true}></Input>
        {Object.keys(credentialsForm).map((key) => { 
          const title = key.charAt(0).toUpperCase() + key.slice(1)
          return (
            <div>
            <FormLabel>{title}</FormLabel>
            <Input name="key" type={credentialsForm[key].type === "CharField" ? "text" : "number"} required={credentialsForm[key].required}></Input>
            </div>
        )})}
        <Button onClick={saveConnection}>
          Save
        </Button>
        </FormControl>
      </Sheet>
    </React.Fragment>
  );
}
