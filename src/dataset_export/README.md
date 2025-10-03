# Importing J1939 Open-Source Dataset

## Prerequisites

* Docker
  ```bash
  # Add Docker's official GPG key:
  sudo apt-get update
  sudo apt-get install ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc
  
  # Add the repository to Apt sources:
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update
  sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  ```
  
* Python >= 3.12


* UV 
  ```bash
  pip install --user uv
  ```

* Download the following [dataset](https://etsin.fairdata.fi/dataset/7586f24f-c91b-41df-92af-283524de8b3e) and extract into the data folder as part_1, part_2, and part_3..
  

## Setup

```bash
uv sync
```

## Run

Starts a postgresDB with docker on port 5435 then runs log import.

```
sudo docker compose up
uv run __main__.py
```
