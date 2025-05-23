# Use official Node.js base image
FROM node:20

# Set working directory
WORKDIR /app

# Copy package.json and lock file
COPY package*.json ./

# Install recharts with legacy-peer-deps
RUN npm install --legacy-peer-deps

# Copy rest of the app
COPY . .

# Expose development port
EXPOSE 3000

# Run development server
CMD ["npm", "run", "dev"]
