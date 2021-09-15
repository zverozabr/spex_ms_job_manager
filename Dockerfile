FROM python:3.9.2
USER root

ENV PYTHONDONTWRITEBYTECODE = 1
ENV PYTHONUNBUFFERED = 1

COPY ./common ./common

WORKDIR /app
COPY ./ms-job-manager /app



RUN pip install pipenv
RUN pip install -e ./../common
RUN pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -e deepcell
RUN pipenv install --system --skip-lock



CMD ["python", "app.py"]
