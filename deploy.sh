#!/bin/bash
gcloud app deploy backend/app.yaml
gcloud app deploy frontend/app.yaml

# to deploy but not actually push live, use --no-promote
# gcloud app deploy --no-promote backend/app.yaml
# gcloud app deploy --no-promote frontend/app.yaml
