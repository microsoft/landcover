#!/bin/sh

git clone https://github.com/microsoft/landcover.git
cd landcover
git checkout feature/vm-arm-template
cd ARM/simple_docker_test

cat << EOF > app/az_config.py
AZ_RESOURCE_GRP="$1"
AZ_RESOURCE_NAME="$2"
EOF

sleep 10m
sudo docker build . -t simple-docker-test:1
sudo docker run --name test --mount type=bind,source="$(pwd)"/logs,target=/app/logs "simple-docker-test:1"
