# Step 1: Use Node.js to build the Angular app
FROM node:18 AS build-stage
WORKDIR ./app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build --prod

# Step 2: Use NGINX to serve the Angular app
FROM nginx:1.25
WORKDIR /usr/share/nginx/html



# Copy built Angular app to NGINX's HTML directory
COPY --from=build-stage /app/dist/frontend/browser /usr/share/nginx/html

# Replace default NGINX configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 (standard for NGINX)
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
