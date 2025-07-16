#!/bin/bash
set -e

# 1. Stop Docker services before changing data-root
sudo systemctl stop docker
sudo systemctl stop docker.socket || true
sudo systemctl stop containerd || true

# 2. Prepare new Docker root directory
sudo mkdir -p /ephemeral/docker

# 3. Move existing Docker data if present
sudo mkdir -p /etc/docker
if [ -f /etc/docker/daemon.json ]; then
  # Merge new key into existing JSON
  sudo jq '. + { "data-root": "/ephemeral/docker" }' /etc/docker/daemon.json | sudo tee /etc/docker/daemon.json.tmp
  sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json
else
  # Create new JSON if file doesn't exist
  echo '{ "data-root": "/ephemeral/docker" }' | sudo tee /etc/docker/daemon.json
fi

# 5. Start Docker with new configuration
sudo systemctl daemon-reload
sudo systemctl start docker

# 6. Install docker-compose-plugin
sudo apt-get update
sudo apt-get install -y docker-compose-plugin

# 7. Install JupyterLab via pip3
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install jupyterlab

# 8. Create workspace directory if it doesn't exist
sudo mkdir -p /ephemeral/workspace

# 9. Launch JupyterLab server with desired options
sudo python3 -m jupyterlab \
  --allow-root \
  --ip=0.0.0.0 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --notebook-dir='/ephemeral/workspace/'
