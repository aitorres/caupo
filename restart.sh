#!/bin/bash
# Restarts both the back-end and the front-end

# Bring latest changes
git pull origin master

# Backend
sudo service backend restart
sudo service nginx restart

# Front-end
cd frontend
npm build
