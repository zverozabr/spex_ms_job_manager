FROM python:3.9.2
USER root

ENV PYTHONDONTWRITEBYTECODE = 1
ENV PYTHONUNBUFFERED = 1

COPY ./microservices/ms-job-manager /app/services/app
COPY ./common /app/common

WORKDIR /app/services/app


RUN pip install pipenv
RUN pip install -e ../../common
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install aicsimageio==4.1.0
RUN pip install tensorflow-cpu
RUN pipenv install --system --skip-lock

CMD ["python", "app.py"]
