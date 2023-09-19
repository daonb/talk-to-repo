# Dockerfile

# Use the official Node.js image as the base
FROM node:19

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the app's source code to the working directory
COPY . .

# Creating an empty corpus_summary.csv file
RUN mkdir -p backend/data
RUN touch backend/data/corpus_summary.csv

# Build the frontend
RUN npm run build

# Expose the port that the frontend will run on
EXPOSE 3000
ENV NEXT_PUBLIC_BACKEND_URL=${NEXT_PUBLIC_BACKEND_URL}

# Start the frontend server
CMD ["npm", "start"]