#!/bin/bash
IMAGE_NAME=streamlit-demo
VERSION_TAG=latest
PROJECT_ID=vertex-project-421816

SERVICE_NAME=streamlit-demo
REGION=us-west1


gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION_TAG
gcloud run deploy $SERVICE_NAME --image gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION_TAG --platform managed --region $REGION --allow-unauthenticated