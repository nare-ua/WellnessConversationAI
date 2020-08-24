ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.03-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/wellness
WORKDIR /workspace/wellness
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 10500
