#!/bin/bash

# Get the port from environment variable (Cloud Run sets PORT)
PORT=${PORT:-80}

# Replace the port in nginx configuration
sed -i "s/listen 80;/listen $PORT;/" /etc/nginx/nginx.conf

# Start nginx
nginx -g "daemon off;"
