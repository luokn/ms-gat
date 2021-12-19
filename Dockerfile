FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

LABEL maintainer="olooook@outlook.com"

RUN apt update && apt install git python3-pip -y 

WORKDIR /

RUN git clone https://github.com/luokn/ms-gat 

RUN pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    pip3 install numpy click pyyaml

COPY ./data /ms-gat/data

WORKDIR /ms-gat

ENTRYPOINT ["python3", "main.py"]
