FROM python:3.9.2
USER root

ENV PYTHONDONTWRITEBYTECODE = 1
ENV PYTHONUNBUFFERED = 1

COPY ./microservices/ms-job-manager /app/services/app
COPY ./common /app/common

WORKDIR /app/services/app


RUN pip install pipenv
RUN pip install -e ../../common
RUN pipenv install --system --skip-lock

CMD ["python", "app.py"]
