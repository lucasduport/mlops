
FROM python:3.12-slim

WORKDIR /code

COPY ./TP1/requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./TP1/app /code/app
COPY ./TP1/regression.joblib /code/regression.joblib

CMD ["fastapi", "run", "app/main.py", "--port", "80"]