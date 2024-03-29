# app/Dockerfile

FROM python:3.11-slim-buster

WORKDIR /CLIC-semseg

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN echo "conda activate myenv" > ~/.bashrc

RUN git clone https://github.com/paulispaulis/CLIC-semseg .

RUN pip3 install -r requirements.txt

WORKDIR /CLIC-semseg/demo

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_demo.py", "--server.port=8501", "--server.address=0.0.0.0"]