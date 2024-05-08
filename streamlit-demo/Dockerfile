FROM python:3.9-slim
WORKDIR /app

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install -r requirements.txt

EXPOSE 8080

ENTRYPOINT [ "sh", "-c", "streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0" ]