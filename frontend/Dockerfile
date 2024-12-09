# frontend/Dockerfile

# Use the official Node.js image
FROM node:22-alpine

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code
COPY . ./

RUN npm run build

# Expose port 5173
EXPOSE 5173

ENV HOST 0.0.0.0

# Start the React development server
CMD ["npm", "run", "dev"]
