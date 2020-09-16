ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:20.08-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace
WORKDIR /workspace

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn gunicorn fastapi pytorch-lightning

#COPY ./start.sh /start.sh
RUN chmod +x /workspace/app/start.sh
RUN chmod +x /workspace/app/start-reload.sh

ENV PYTHONPATH=/workspace/app

ENV PORT=8000
EXPOSE 8000

CMD ["/workspace/app/start.sh"]
