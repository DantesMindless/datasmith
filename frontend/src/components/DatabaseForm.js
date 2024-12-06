import React, { useEffect, useState } from "react";
import axios from "../axiosConfig.js";
import { useForm } from "react-hook-form";

const DataBaseForm = () => {
    const [dataSources, setDataSources] = useState([]);
    const [selectedDataSource, setSelectedDataSource] = useState(null);
    const [queryResult, setQueryResult] = useState(null);
    const { register, handleSubmit, reset } = useForm();

    // Fetch the list of data sources
    useEffect(() => {
        axios
            .get("/datasource/")
            .then((response) => {
                setDataSources(response.data);
            })
            .catch((error) => {
                console.error("Error fetching data sources:", error);
            });
    }, []);

    const handleQuery = async (data) => {
        if (!selectedDataSource) {
            alert("Please select a data source first!");
            return;
        }

        try {
            const response = await axios.put(`/datasource/query/${selectedDataSource}/`, {
                query: data.query,
            });
            setQueryResult(response.data);
            alert("Query executed successfully!");
        } catch (error) {
            console.error("Error executing query:", error);
            alert("Query execution failed!");
        }
    };

    // Test Connection Handler
    const handleTestConnection = async (data) => {
        try {
            const response = await axios.post("/datasource/test/", {
                name: data.name,
                type: data.type,
                description: data.description,
                credentials: {
                    host: data.credentials.host,
                    port: data.credentials.port,
                    database: data.credentials.database,
                    user: data.credentials.user,
                    password: data.credentials.password,
                },
            });
            alert("Connection successful: " + response.data.message);
        } catch (error) {
            console.error("Error testing connection:", error);
            if (error.response && error.response.data.error) {
                alert("Connection failed: " + error.response.data.error);
            } else {
                alert("An unexpected error occurred.");
            }
        }
    };


    // Store Connection Handler
    const handleStoreConnection = (data) => {
        axios
            .post("/datasource/", data)
            .then((response) => {
                alert("Connection stored successfully!");
                setDataSources((prev) => [...prev, response.data]); // Update list
            })
            .catch((error) => {
                console.error("Error storing connection:", error);
                alert("Failed to store connection!");
            });
    };

    // Connect Handler
    const handleConnect = () => {
        if (!selectedDataSource) {
            alert("Please select a data source first!");
            return;
        }
        alert("Attempting to connect to " + selectedDataSource);
        // Add your backend-specific connect logic here
    };

    // Cancel Handler
    const handleCancel = () => {
        reset(); // Reset the form fields
        setSelectedDataSource(null); // Clear selection
        alert("Operation cancelled!");
    };

    return (
        <div className="container">
            <h1>Data Source Management</h1>

            {/* List Data Sources */}
            <section>
                <h2>Available Data Sources</h2>
                <ul>
                    {dataSources.map((ds) => (
                        <li key={ds.id}>
                            {ds.name} ({ds.type})
                            <button
                                onClick={() => setSelectedDataSource(ds.id)}
                                className={selectedDataSource === ds.id ? "btn btn-primary" : "btn btn-secondary"}
                            >
                                Select
                            </button>
                        </li>
                    ))}
                </ul>
            </section>

            {/* Test Connection Form */}
            <section>
                <h2>Test Data Source Connection</h2>
                <form onSubmit={handleSubmit(handleTestConnection)}>
                    <div>
                        <label>Data Source Name</label>
                        <input
                            {...register("name", { required: true })}
                            type="text"
                            placeholder="Data Source Name"
                        />
                    </div>
                    <div>
                        <label>Type</label>
                        <select {...register("type", { required: true })}>
                            <option value="POSTGRES">PostgreSQL</option>
                            <option value="MYSQL">MySQL</option>
                            <option value="sqlite">SQLite</option>
                        </select>
                    </div>
                    
                    <div>
                        <label>Host</label>
                        <input
                            {...register("credentials.host", { required: true })}
                            type="text"
                            placeholder="Host"
                        />
                    </div>
                    <div>
                        <label>Port</label>
                        <input
                            {...register("credentials.port", { required: true })}
                            type="number"
                            placeholder="Port"
                        />
                    </div>
                    <div>
                        <label>Description</label>
                        <input
                            {...register("description", { required: true })}
                            type="text"
                            placeholder="Description"
                        />
                    </div>
                    <div>
                        <label>Database</label>
                        <input
                            {...register("credentials.database", { required: true })}
                            type="text"
                            placeholder="Database"
                        />
                    </div>
                    <div>
                        <label>Username</label>
                        <input
                            {...register("credentials.user", { required: true })}
                            type="text"
                            placeholder="Username"
                        />
                    </div>
                    <div>
                        <label>Password</label>
                        <input
                            {...register("credentials.password", { required: true })}
                            type="password"
                            placeholder="Password"
                        />
                    </div>

                    {/* Buttons for Different Actions */}
                    <div className="button-group">
                        <button type="button" onClick={handleCancel}>
                            Cancel
                        </button>
                        <button type="button" onClick={handleSubmit(handleStoreConnection)}>
                            Store
                        </button>
                        <button type="button" onClick={handleSubmit(handleTestConnection)}>
                            Test
                        </button>
                        <button type="button" onClick={handleConnect}>
                            Connect
                        </button>
                    </div>
                </form>
            </section>

            {/* Query Form */}
            <section>
                <h2>Run a Query</h2>
                {selectedDataSource ? (
                    <form onSubmit={handleSubmit(handleQuery)}>
                        <div>
                            <label>SQL Query</label>
                            <textarea
                                {...register("query", { required: true })}
                                placeholder="Write your SQL query here..."
                            ></textarea>
                        </div>
                        <button type="submit">Run Query</button>
                    </form>
                ) : (
                    <p>Please select a data source to run a query.</p>
                )}
            </section>

            {/* Query Results */}
            {queryResult && (
                <section>
                    <h2>Query Results</h2>
                    <pre>{JSON.stringify(queryResult, null, 2)}</pre>
                </section>
            )}
        </div>
    );
};

export default DataBaseForm;
