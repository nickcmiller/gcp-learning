FROM python:3.9-slim

WORKDIR /app

# Copy local files into the Docker image
COPY ./requirements.txt ./requirements.txt
COPY ./streamlit_app.py ./streamlit_app.py

RUN pip3 install -r requirements.txt

EXPOSE $PORT

HEALTHCHECK CMD curl --fail http://localhost:$PORT/_stcore/health

ENTRYPOINT streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0