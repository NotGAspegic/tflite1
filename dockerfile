FROM python:3.9.18


WORKDIR /tflite

COPY requirements.txt .
COPY ./src ./src

RUN pip install -r requirements.txt


CMD [ "python","./src/Stream.py" ]

