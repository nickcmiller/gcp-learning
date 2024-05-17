FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy local files into the Docker image
COPY ./requirements.txt ./requirements.txt
COPY ./streamlit_langchain_app.py ./streamlit_langchain_app.py

# Install dependencies
RUN pip3 install -r requirements.txt

# Cloud Run sets $PORT
# The default port used by Cloud Run is 8080
EXPOSE $PORT

# Set the health check to check that container is running
HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health

# Run the streamlit command to start the streamlit application
ENTRYPOINT streamlit run streamlit_langchain_app.py --server.port=$PORT --server.address=0.0.0.0