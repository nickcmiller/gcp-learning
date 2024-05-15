#!/bin/bash
IMAGE_NAME=streamlit-demo
VERSION_TAG=latest
PROJECT_ID=vertex-project-421816

SERVICE_NAME=streamlit-demo
REGION=us-west1
# Hide the GROQ_API_KEY from the CLI using `export GROQ_API_KEY=your-api-key`
GROQ_API_KEY=$GROQ_API_KEY

gcloud builds submit --tag gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION_TAG
gcloud run deploy $SERVICE_NAME --image gcr.io/$PROJECT_ID/$IMAGE_NAME:$VERSION_TAG --platform managed --region $REGION --allow-unauthenticated --set-env-vars=GROQ_API_KEY=$GROQ_API_KEY