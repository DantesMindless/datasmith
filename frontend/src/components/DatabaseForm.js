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
            <div className="row">

                <h1>Data Source Management</h1>

                {/* List Data Sources */}


                {/* Test Connection Form */}
                <section className="col-7 text-center">
                    <h2>Test Data Source Connection</h2>
                    <div className="d-flex bd-highlight">
                        <form onSubmit={handleSubmit(handleTestConnection)}>
                            <div className="m-4 d-flex">
                                <label className="bd-highlight">Name</label>
                                <input className="ms-auto"
                                    {...register("name", { required: true })}
                                    type="text"
                                    placeholder="Data Source Name"
                                />
                            </div>
                            <div className="m-4 d-flex">
                                <label className="bd-highlight">Type</label>
                                <select className="ms-auto" {...register("type", { required: true })}>
                                    <option value="POSTGRES">PostgreSQL</option>
                                    <option value="MYSQL">MySQL</option>
                                    <option value="SQLITE">SQLite</option>
                                </select>
                            </div>

                            <div className="m-4 d-flex">
                                <label className="bd-highlight">Host</label>
                                <input className="ms-auto"
                                    {...register("credentials.host", { required: true })}
                                    type="text"
                                    placeholder="Host"
                                />
                            </div>
                            <div className="m-4 d-flex">
                                <label className="bd-highlight">Port</label>
                                <input className="ms-auto"
                                    {...register("credentials.port", { required: true })}
                                    type="number"
                                    placeholder="Port"
                                />
                            </div>
                            <div className="m-4 d-flex">
                                <label className="bd-highlight">Description</label>
                                <input className="ms-auto"
                                    {...register("description", { required: true })}
                                    type="text"
                                    placeholder="Description"
                                />
                            </div>
                            <div className="m-4 d-flex">
                                <label className="bd-highlight">Database</label>
                                <input className="ms-auto"
                                    {...register("credentials.database", { required: true })}
                                    type="text"
                                    placeholder="Database"
                                />
                            </div>
                            <div className="m-4 d-flex">
                                <label className=" bd-highlight">Username</label>
                                <input className="ms-auto"
                                    {...register("credentials.user", { required: true })}
                                    type="text"
                                    placeholder="Username"
                                />
                            </div>
                            <div className="m-4 d-flex">
                                <label className=" bd-highlight">Password</label>
                                <input className="ms-auto"
                                    {...register("credentials.password", { required: true })}
                                    type="password"
                                    placeholder="Password"
                                />
                            </div>

                            {/* Buttons for Different Actions */}
                            <div className="button-group m-4">
                                <button className="m-2" type="button" onClick={handleCancel}>
                                    Cancel
                                </button>
                                <button className="m-2" type="button" onClick={handleSubmit(handleStoreConnection)}>
                                    Store
                                </button>
                                <button className="m-2" type="button" onClick={handleSubmit(handleTestConnection)}>
                                    Test
                                </button>
                                <button className="m-2" type="button" onClick={handleConnect}>
                                    Connect
                                </button>
                            </div>
                        </form>
                    </div>
                </section>
                <section className="col-5 ">
                    <h2>Available Data Sources</h2>
                    <ul className="list-unstyled">
                        {dataSources.map((ds) => (
                            <li key={ds.id} className="d-flex bd-highlight">
                                <div className="text-start flex-grow-1 bd-highlight">{ds.name} ({ds.type})</div>
                                <button
                                    onClick={() => setSelectedDataSource(ds.id)}
                                    className={selectedDataSource === ds.id ? "m-2 btn btn-primary" : "m-2 btn btn-secondary"}
                                >
                                    Select
                                </button>
                            </li>
                        ))}
                    </ul>
                </section>
                {/* Query Form */}
                <h1 className="m-2">Run a Query</h1>
                <section className="col-12 d-flex justify-content-center">

                    {selectedDataSource ? (
                        <form onSubmit={handleSubmit(handleQuery)}>
                            <div>

                                <textarea className="m-3" style={{
                                    minWidth: "700px",
                                    maxWidth: "100%",
                                    minHeight: "50px",
                                    height: "100%",
                                    width: "100%",
                                }}
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
        </div>
    );
};

export default DataBaseForm;
