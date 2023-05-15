# Use an official Node runtime as a parent image
FROM node:14-alpine

# Set the working directory to /app
WORKDIR /app

# copy the package.json to cache the required npm libraries
COPY package*.json ./

# Install any needed packages
RUN npm install

# INSTALL VIM
RUN apt-get update && apt-get install -y vim

# Copy aliases
COPY .bash_aliases /root/
RUN echo "if [ -f /root/.bash_aliases ]; then . /root/.bash_aliases; fi" >> /root/.bashrc

# Copy the current directory contents into the container at /app
COPY . .

# Build the app
RUN npm run build

# Install serve globally
RUN npm install -g serve

# Expose the port the app runs on
EXPOSE 3000

# Set the command to start the app
CMD ["serve", "-s", "build"]


