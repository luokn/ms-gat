FROM luokn/pytorch-runtime

LABEL maintainer="luokun485@gmail.com"

WORKDIR /ms-gat

COPY ./ /ms-gat/

ENTRYPOINT ["python3", "src/main.py"]
