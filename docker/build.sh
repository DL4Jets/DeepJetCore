#!/usr/bin/bash

docker build -t cernml4reco/deepjetcore3:latest . --build-arg BUILD_DATE="$(date)"