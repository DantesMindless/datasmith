import React, { useEffect, useState } from 'react';

const HelloWorld = () => {
    const [message, setMessage] = useState('');

    useEffect(() => {
        // Fetch the message from the Django API endpoint
        const fetchMessage = async () => {
            try {
                const response = await fetch(process.env.REACT_APP_API_URL + "hello");
                const data = await response.json();
                setMessage(data.message);
            } catch (error) {
                console.error("Error fetching message:", error);
            }
        };

        fetchMessage();
    }, []);

    return (
        <div>
            <h1>{message}</h1>
        </div>
    );
};

export default HelloWorld;
