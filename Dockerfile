FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04

LABEL maintainer="olooook@outlook.com"

RUN apt update && apt install python3-pip git -y

RUN pip3 install numpy click pyyaml

RUN pip3 install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

WORKDIR /ms-gat

COPY ./ /ms-gat/

ENTRYPOINT ["python3", "src/main.py"]
